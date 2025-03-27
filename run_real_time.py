import subprocess
import time
import sys

def run_system():
    print("Starting server...")
    flask_process = subprocess.Popen([sys.executable, "real_time_user_study.py"])
    
    print("Waiting for Flask server to start...")
    time.sleep(3)
    
    print("Starting prediction modle...")
    prediction_process = subprocess.Popen([sys.executable, "real_time_prediction.py"])
    
    try:
        # 保持脚本运行
        flask_process.wait()
        prediction_process.wait()
    except KeyboardInterrupt:
        print("Stopping all processes...")
        flask_process.terminate()
        prediction_process.terminate()
        flask_process.wait()
        prediction_process.wait()

if __name__ == "__main__":
    run_system()


# 先别用 二合一还没写完