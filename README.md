# searchpad-api

Running the FastAPI Server

Using uvicorn
uvicorn is an ASGI server for Python, ideal for serving FastAPI applications.

```bash
uvicorn app.main:app --reload
```

Explanation:

app.main:app tells uvicorn to find the app object in app/main.py.
--reload enables auto-reloading on code changes (useful during development).

Full Command with Host and Port:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Host 0.0.0.0: Makes the server accessible externally (e.g., on a local network).
Port 8000: You can change this to any available port.
