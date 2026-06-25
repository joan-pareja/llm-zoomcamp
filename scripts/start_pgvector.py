import subprocess


def main() -> None:
    subprocess.run(
        [
            "docker",
            "run",
            "-it",
            "--name",
            "pgvector",
            "-e",
            "POSTGRES_USER=user",
            "-e",
            "POSTGRES_PASSWORD=pswd",
            "-e",
            "POSTGRES_DB=faq",
            "-v",
            "pgvector_data:/var/lib/postgresql/data",
            "-p",
            "5432:5432",
            "pgvector/pgvector:pg17",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
