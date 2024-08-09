import urllib.parse
import psutil
from TextClassification.Inference.infer import Infer
from prometheus_client import Gauge, CollectorRegistry, generate_latest
from prometheus_client import start_http_server
import time

# Initialize the inference class
inference_routes = Infer()

# Create a new registry to hold the metrics
registry = CollectorRegistry()

# Define system metrics
CPU_USAGE = Gauge('system_cpu_usage', 'System CPU usage percentage', registry=registry)
MEMORY_USAGE = Gauge('system_memory_usage', 'System memory usage percentage', registry=registry)
DISK_USAGE = Gauge('system_disk_usage', 'System disk usage percentage', registry=registry)
NETWORK_SENT = Gauge('system_network_sent_bytes', 'Bytes sent over the network', registry=registry)
NETWORK_RECEIVED = Gauge('system_network_received_bytes', 'Bytes received over the network', registry=registry)

# Define custom metrics
SCORE = Gauge('inference_score', 'Inference score', registry=registry)
INFER_TIME = Gauge('inference_time', 'Inference time', registry=registry)

def get_predictions_text_classification(data):
    project_id = data.get('project_id')
    modelID = data.get('modelID')
    device_name = data.get('device_name')
    encoded_input = data.get('input')

    if not project_id or not modelID or not device_name or not encoded_input:
        return 'Missing input parameters', 400

    input_text = urllib.parse.unquote(encoded_input)
    result = inference_routes.get_predictions_text_classification(project_id, modelID, device_name, input_text)

    # Set the custom metrics
    score = result['result'][0]['score']
    SCORE.set(score)
    infer_time = result['result'][0]['infer_time']
    INFER_TIME.set(infer_time)

    return result

def collect_system_metrics():
    # Collect CPU usage
    cpu_usage = psutil.cpu_percent(interval=1)
    CPU_USAGE.set(cpu_usage)

    # Collect memory usage
    memory_info = psutil.virtual_memory()
    MEMORY_USAGE.set(memory_info.percent)

    # Collect disk usage
    disk_usage = psutil.disk_usage('/')
    DISK_USAGE.set(disk_usage.percent)

    # Collect network statistics
    net_io = psutil.net_io_counters()
    NETWORK_SENT.set(net_io.bytes_sent)
    NETWORK_RECEIVED.set(net_io.bytes_recv)

def start_metrics_server():
    # Start the HTTP server on port 8000
    start_http_server(8000, registry=registry)
    print("Metrics server started at http://localhost:8000")

# Example usage in a Jupyter Notebook
if __name__ == '__main__':
    # Start the metrics server
    start_metrics_server()

    # Place the function call here to test it
    data = {
        "project_id": "2185159c-8079-42a1-89dc-f3818e14816b",
        "modelID": "facebook/bart-large-mnli",
        "device_name": "SapphireRapidsCPU",
        'input': urllib.parse.quote('I loved star wars so much')  # URL encode the input text
    }

    # Call the function with the example data
    result = get_predictions_text_classification(data)

    # Print the result
    print(result)

    # Run a loop to collect system metrics periodically
    while True:
        collect_system_metrics()
        time.sleep(5)  # Adjust the sleep time as needed

