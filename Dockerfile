FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy Code
COPY server.py client.py cnn_model.py /app/

# COPY DATA (The critical part)
COPY data /app/data

# --- THE TRAP ---
# This command lists the file during the build. 
# IF THIS FAILS, THE BUILD CRASHES HERE.
RUN ls -l /app/data/client_0/round_1.csv

CMD ["python", "-u", "client.py"]
