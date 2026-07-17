run:
	uv run python ./05-monitor/assistant.py

chat:
	uv run streamlit run ./05-monitor/app.py --server.port 8501

dashboard:
	uv run streamlit run ./05-monitor/dashboard.py --server.port 8502

generate-data:
	uv run python ./05-monitor/generate_synthetic_data.py

monitor-db-up:
	docker compose --env-file ./.env -f ./05-monitor/docker-compose.yml up -d

monitor-db-down:
	docker compose -f ./05-monitor/docker-compose.yml down

monitor-db-init:
	uv run python ./05-monitor/initialize_monitoring_database.py
