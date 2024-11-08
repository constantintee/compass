# docker-entrypoint.sh
#!/bin/sh

echo "Waiting for postgres..."
while ! nc -z $DB_HOST $DB_PORT; do
    echo "Postgres is unavailable - sleeping"
    sleep 1
done
echo "Postgres is up - executing command"

# Run migrations
python manage.py migrate

# Collect static files
python manage.py collectstatic --no-input

# Start application
exec "$@"