import pandas as pd
import joblib
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import io
import os
import json

app = Flask(__name__)
CORS(app)

# Resolve paths relative to this file so the app works from any CWD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def _model_path(filename: str) -> str:
	return os.path.join(MODELS_DIR, filename)

# Load all 5 models into memory
churn_model = joblib.load(_model_path('churn_model.pkl'))
sales_forecast_model = joblib.load(_model_path('sales_forecast_model.pkl'))
clv_model = joblib.load(_model_path('clv_model.pkl'))
review_model = joblib.load(_model_path('review_model.pkl'))
delivery_model = joblib.load(_model_path('delivery_model.pkl'))

# Helper: map model name to object and output column
def get_model_and_output_column(model_name: str):
	name = (model_name or '').strip().lower()
	if name == 'churn':
		return churn_model, 'churn_prediction'
	if name == 'clv':
		return clv_model, 'predicted_clv_90_days'
	if name == 'review' or name == 'review_score':
		return review_model, 'review_prediction'
	if name == 'delivery' or name == 'delivery_time':
		return delivery_model, 'delivery_time_prediction_days'
	return None, None

# Helper: best-effort preprocessing to match model's expected features and dtypes
def _derive_clv_features(df: pd.DataFrame) -> pd.DataFrame:
	# Use explicit column names from provided CSV header
	customer_col = 'customer_id'
	date_col = 'order_purchase_timestamp'
	value_col = 'payment_value'
	for required in [customer_col, date_col, value_col]:
		if required not in df.columns:
			raise ValueError('Cannot derive CLV features; missing column: ' + required)

	# If CLV features already exist, return a minimal frame for merge consistency
	existing = ['frequency_cal', 'recency_cal', 'T_cal', 'monetary_value_cal']
	if all(col in df.columns for col in existing):
		return df[[customer_col] + existing].copy(), customer_col

	work = df[[customer_col, date_col, value_col]].copy()
	work[date_col] = pd.to_datetime(work[date_col], errors='coerce', utc=True)
	work[value_col] = pd.to_numeric(work[value_col], errors='coerce')
	work = work.dropna(subset=[date_col])

	# Use max observed date in the dataset as calibration end
	calibration_end = work[date_col].max()

	# Aggregate per customer
	grp = work.sort_values([customer_col, date_col]).groupby(customer_col, as_index=False)
	agg = grp.agg(
		first_purchase=(date_col, 'first'),
		last_purchase=(date_col, 'last'),
		num_tx=(date_col, 'count'),
		avg_monetary=(value_col, 'mean')
	)
	agg['frequency_cal'] = (agg['num_tx'] - 1).clip(lower=0)
	agg['recency_cal'] = (agg['last_purchase'] - agg['first_purchase']).dt.total_seconds() / 86400.0
	agg['T_cal'] = (calibration_end - agg['first_purchase']).dt.total_seconds() / 86400.0
	agg['monetary_value_cal'] = agg['avg_monetary']

	result = agg[[customer_col, 'frequency_cal', 'recency_cal', 'T_cal', 'monetary_value_cal']].copy()
	return result, customer_col

def preprocess_for_model(input_df: pd.DataFrame, model_name: str, model) -> pd.DataFrame:
	processed = input_df.copy()
	# If model exposes expected feature names, align to them and order columns
	if hasattr(model, 'feature_names_in_'):
		feature_names = list(model.feature_names_in_)
		missing = [c for c in feature_names if c not in processed.columns]
		# Special handling for CLV: derive RFM calibration features if missing
		if missing and (model_name or '').lower() == 'clv':
			needed = {'frequency_cal', 'recency_cal', 'T_cal', 'monetary_value_cal'}
			if any(col in needed for col in missing):
				try:
					clv_feats, customer_key = _derive_clv_features(processed)
					processed = processed.merge(clv_feats, how='left', on=customer_key)
					# Recompute missing after merge
					missing = [c for c in feature_names if c not in processed.columns]
				except Exception as e:
					raise ValueError(str(e))
		if missing:
			raise ValueError('Missing required feature columns: ' + ', '.join(missing))
		processed = processed[feature_names]

	# Convert obvious datetime-like columns to numeric timestamps if dtype is object
	for col in processed.columns:
		series = processed[col]
		if pd.api.types.is_datetime64_any_dtype(series):
			processed[col] = series.view('int64') // 10**9
			continue
		if series.dtype == 'object':
			# Try parse datetime
			parsed = pd.to_datetime(series, errors='coerce', utc=True)
			if parsed.notna().any() and parsed.isna().sum() < len(parsed):
				processed[col] = parsed.view('int64') // 10**9
				continue
			# Try numeric
			num = pd.to_numeric(series, errors='coerce')
			if num.notna().any():
				processed[col] = num
				continue
			# Fallback to categorical codes (note: codes may differ from training encoders)
			processed[col] = series.astype('category').cat.codes

	# Ensure all remaining non-numeric types are cast to numeric where possible
	for col in processed.columns:
		if not pd.api.types.is_numeric_dtype(processed[col]):
			processed[col] = pd.to_numeric(processed[col], errors='coerce').fillna(0)

	return processed

# Helper: run predictions on a DataFrame for a given model
def predict_dataframe(model_name: str, df: pd.DataFrame) -> pd.Series:
	model, output_col = get_model_and_output_column(model_name)
	if model is None:
		raise ValueError('Unknown model: ' + str(model_name))
	# Align and preprocess input features to what the model expects
	prepared_df = preprocess_for_model(df, model_name, model)
	# Models are trained for single-row predictions but work with batch DataFrames too
	preds = model.predict(prepared_df)
	return pd.Series(preds, index=df.index, name=output_col)

@app.route('/api/predict/churn', methods=['POST'])
def predict_churn():
    data = request.get_json(force=True)
    df = pd.DataFrame(data, index=[0])
    prediction = churn_model.predict(df)
    return jsonify({'churn_prediction': int(prediction[0])})

@app.route('/api/forecast/sales', methods=['GET'])
def forecast_sales():
    forecast = sales_forecast_model.predict(n_periods=90)
    return jsonify({'sales_forecast': forecast.tolist()})

@app.route('/api/predict/clv', methods=['POST'])
def predict_clv():
    data = request.get_json(force=True)
    df = pd.DataFrame(data, index=[0])
    prediction = clv_model.predict(df)
    return jsonify({'predicted_clv_90_days': round(prediction[0], 2)})

@app.route('/api/predict/review_score', methods=['POST'])
def predict_review():
    # Get data from the POST request
    data = request.get_json(force=True)
    # Convert data into DataFrame
    df = pd.DataFrame(data, index=[0])
    
    # Make prediction using the loaded review_model
    prediction = review_model.predict(df)
    
    # Return prediction as JSON
    return jsonify({'review_prediction': int(prediction[0])})

@app.route('/api/predict/delivery_time', methods=['POST'])
def predict_delivery():
    # Get data from the POST request
    data = request.get_json(force=True)
    # Convert data into DataFrame
    df = pd.DataFrame(data, index=[0])
    
    # Make prediction using the loaded delivery_model
    prediction = delivery_model.predict(df)
    
    # Return prediction as JSON, rounded to 2 decimal places
    return jsonify({'delivery_time_prediction_days': round(prediction[0], 2)})

# Batch prediction endpoint: accepts CSV upload or JSON array of records
@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
	model_name = request.args.get('model')
	response_format = (request.args.get('format') or 'json').lower()
	if not model_name:
		return jsonify({'error': 'Missing required query param: model'}), 400

	# Accept either file upload (multipart/form-data) or JSON array
	if 'file' in request.files:
		file_storage = request.files['file']
		try:
			# Let pandas infer CSV/Excel by extension; default to CSV
			filename = (file_storage.filename or '').lower()
			if filename.endswith('.xlsx') or filename.endswith('.xls'):
				df = pd.read_excel(file_storage)
			else:
				df = pd.read_csv(file_storage)
		except Exception as e:
			return jsonify({'error': 'Failed to parse file', 'details': str(e)}), 400
	else:
		# Try JSON; expect list of records or object with key "records"
		try:
			payload = request.get_json(force=True)
			records = payload.get('records') if isinstance(payload, dict) else payload
			if not isinstance(records, list):
				return jsonify({'error': 'Expected JSON array or {"records": [...]}'}), 400
			df = pd.DataFrame.from_records(records)
		except Exception as e:
			return jsonify({'error': 'Failed to parse JSON body', 'details': str(e)}), 400

	try:
		pred_series = predict_dataframe(model_name, df)
		result_df = df.copy()
		# Round numeric outputs as per single-row endpoints
		if pred_series.name == 'delivery_time_prediction_days' or pred_series.name == 'predicted_clv_90_days':
			pred_series = pred_series.astype(float).round(2)
		if pred_series.name == 'churn_prediction' or pred_series.name == 'review_prediction':
			pred_series = pred_series.astype(int)
		result_df[pred_series.name] = pred_series.values
	except Exception as e:
		return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

	if response_format == 'csv':
		csv_buffer = io.StringIO()
		result_df.to_csv(csv_buffer, index=False)
		response = make_response(csv_buffer.getvalue())
		response.headers['Content-Type'] = 'text/csv'
		response.headers['Content-Disposition'] = 'attachment; filename="predictions.csv"'
		return response
	# default JSON
	return jsonify({
		'model': model_name,
		'num_rows': int(len(result_df)),
		'predictions_column': pred_series.name,
		'data': result_df.to_dict(orient='records')
	})

# File-based prediction on a server-side dataset path, for BI connectors that use GET
@app.route('/api/predict/file', methods=['GET'])
def predict_from_file():
	model_name = request.args.get('model')
	data_path = request.args.get('path')
	response_format = (request.args.get('format') or 'json').lower()
	if not model_name or not data_path:
		return jsonify({'error': 'Missing required query params: model, path'}), 400
	if not os.path.exists(data_path):
		return jsonify({'error': 'File not found', 'path': data_path}), 404
	try:
		if data_path.lower().endswith('.xlsx') or data_path.lower().endswith('.xls'):
			df = pd.read_excel(data_path)
		else:
			df = pd.read_csv(data_path)
		pred_series = predict_dataframe(model_name, df)
		result_df = df.copy()
		if pred_series.name == 'delivery_time_prediction_days' or pred_series.name == 'predicted_clv_90_days':
			pred_series = pred_series.astype(float).round(2)
		if pred_series.name == 'churn_prediction' or pred_series.name == 'review_prediction':
			pred_series = pred_series.astype(int)
		result_df[pred_series.name] = pred_series.values
	except Exception as e:
		return jsonify({'error': 'Prediction failed', 'details': str(e), 'path': data_path}), 500

	if response_format == 'csv':
		csv_buffer = io.StringIO()
		result_df.to_csv(csv_buffer, index=False)
		response = make_response(csv_buffer.getvalue())
		response.headers['Content-Type'] = 'text/csv'
		response.headers['Content-Disposition'] = 'attachment; filename="predictions.csv"'
		return response
	return jsonify({
		'model': model_name,
		'num_rows': int(len(result_df)),
		'predictions_column': pred_series.name,
		'data': result_df.to_dict(orient='records')
	})

@app.route('/api/models', methods=['GET'])
def list_models():
	return jsonify({
		'models': [
			{'name': 'churn', 'endpoint': '/api/predict/churn', 'batch': '/api/predict/batch?model=churn'},
			{'name': 'clv', 'endpoint': '/api/predict/clv', 'batch': '/api/predict/batch?model=clv'},
			{'name': 'review', 'endpoint': '/api/predict/review_score', 'batch': '/api/predict/batch?model=review'},
			{'name': 'delivery', 'endpoint': '/api/predict/delivery_time', 'batch': '/api/predict/batch?model=delivery'},
			{'name': 'sales_forecast', 'endpoint': '/api/forecast/sales', 'note': 'time series only'}
		]
	})

@app.route('/health', methods=['GET'])
def health():
	return jsonify({'status': 'ok'})

# Combined predictions: run multiple models over the same dataset and return a merged result
@app.route('/api/predict/file/all', methods=['GET'])
def predict_all_from_file():
	data_path = request.args.get('path')
	# models param can be comma-separated list, default to all row-wise models
	models_param = (request.args.get('models') or 'clv,churn,review,delivery')
	requested_models = [m.strip().lower() for m in models_param.split(',') if m.strip()]
	response_format = (request.args.get('format') or 'csv').lower()
	if not data_path:
		return jsonify({'error': 'Missing required query param: path'}), 400
	if not os.path.exists(data_path):
		return jsonify({'error': 'File not found', 'path': data_path}), 404

	try:
		if data_path.lower().endswith('.xlsx') or data_path.lower().endswith('.xls'):
			df = pd.read_excel(data_path)
		else:
			df = pd.read_csv(data_path)
		result_df = df.copy()
		added_cols = []
		errors = {}
		for m in requested_models:
			try:
				pred_series = predict_dataframe(m, df)
				# Round/cast per model conventions
				if pred_series.name in ('delivery_time_prediction_days', 'predicted_clv_90_days'):
					pred_series = pred_series.astype(float).round(2)
				if pred_series.name in ('churn_prediction', 'review_prediction'):
					pred_series = pred_series.astype(int)
				result_df[pred_series.name] = pred_series.values
				added_cols.append(pred_series.name)
			except Exception as e:
				# Ensure a placeholder column exists even on failure
				_, out_col = get_model_and_output_column(m)
				if out_col and out_col not in result_df.columns:
					result_df[out_col] = pd.NA
				errors[m] = str(e)
	except Exception as e:
		return jsonify({'error': 'Failed to process file', 'details': str(e)}), 400

	# If none succeeded, return errors
	if not added_cols:
		return jsonify({'error': 'Prediction failed for all requested models', 'details': errors}), 500

	if response_format == 'json':
		return jsonify({
			'models': requested_models,
			'num_rows': int(len(result_df)),
			'prediction_columns': added_cols,
			'errors': errors,
			'data': result_df.to_dict(orient='records')
		})
	# default CSV
	csv_buffer = io.StringIO()
	result_df.to_csv(csv_buffer, index=False)
	response = make_response(csv_buffer.getvalue())
	response.headers['Content-Type'] = 'text/csv'
	response.headers['Content-Disposition'] = 'attachment; filename="predictions_all.csv"'
	if errors:
		response.headers['X-Model-Errors'] = json.dumps(errors)
	return response

if __name__ == '__main__':
    app.run(debug=True, port=5000)