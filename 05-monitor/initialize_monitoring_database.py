"""Create the monitoring database schema."""

import sys
from pathlib import Path

from psycopg.errors import ConnectionTimeout

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    from lib.monitoring_store import MonitoringDatabaseSettings, MonitoringStore

    settings = MonitoringDatabaseSettings.from_dotenv()

    try:
        store = MonitoringStore(connection=settings.connect())
    except ConnectionTimeout as exc:
        raise RuntimeError(
            "Could not reach the monitoring Postgres database. "
            "Start it with `make monitor-db-up`, wait until the container is ready, "
            f"and confirm {settings.host}:{settings.port} is reachable."
        ) from exc

    try:
        store.initialize_schema()
    finally:
        store.close()


if __name__ == "__main__":
    main()
