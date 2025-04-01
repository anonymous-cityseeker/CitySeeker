from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS
import json
import os

app = Flask(__name__)

# 配置允许的城市，确保与实际文件夹名称一致
ALLOWED_CITIES = [
    'BeiJing_1', 'ShangHai_2', 'ShenZhen_3',
    'ChengDu_4', 'HongKong_5', 'HongKong_6',
    'NewYork_8', 'London_7'
]

# 配置 CORS 规则，允许所有来源访问特定路由
CORS(app, resources={
    r"/log": {"origins": "*"},
    r"/trace": {"origins": "*"},
    # r"/<city>/trace": {"origins": "*"},
    # r"/<city>/log": {"origins": "*"},
    r"/<city>/connections.json": {"origins": "*"},
    r"/<city>/question.json": {"origins": "*"},
    r"/<city>/<path:filename>": {"origins": "*"},
    r"/viewer": {"origins": "*"},
    r"/static/<path:filename>": {"origins": "*"}
})

LOG_FILE = 'interaction_logs.json'
TRACE_FILE = 'trace.json'

# 确保日志文件存在
for file in [LOG_FILE, TRACE_FILE]:
    if not os.path.exists(file):
        with open(file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)

@app.route('/viewer')
def viewer_route():
    return send_file('viewer.html')

@app.route('/<city>/connections.json')
def connections(city):
    print(f"Received request for connections.json of city: {city}")
    if city not in ALLOWED_CITIES:
        print(f"City not allowed: {city}")
        return jsonify({"error": "城市不被支持"}), 404
    file_path = os.path.join(city, 'connections.json')
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return jsonify({"error": "connections.json 文件不存在"}), 404
    return send_file(file_path)

@app.route('/<city>/question.json')
def question(city):
    print(f"Received request for question.json of city: {city}")
    if city not in ALLOWED_CITIES:
        print(f"City not allowed: {city}")
        return jsonify({"error": "城市不被支持"}), 404
    file_path = os.path.join(city, 'question.json')
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return jsonify({"error": "question.json 文件不存在"}), 404
    return send_file(file_path)

@app.route('/<city>/<path:filename>')
def serve_city_file(city, filename):
    print(f"Received request for file: {filename} in city: {city}")
    if city not in ALLOWED_CITIES:
        print(f"City not allowed: {city}")
        return jsonify({"error": "城市不被支持"}), 404
    # 构建文件路径
    file_path = os.path.join(city, filename)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return jsonify({"error": "文件不存在"}), 404
    # 发送文件
    try:
        return send_file(file_path)
    except Exception as e:
        print(f"发送文件时出错: {e}")
        return jsonify({"error": "服务器内部错误"}), 500

# @app.route('/trace', methods=['POST'])
# def save_trace():
#     data = request.get_json()
#     if not data:
#         return jsonify({"error": "没有提供数据"}), 400
    
#     # 添加数据验证
#     required_fields = ['question_idx', 'timesteps']
#     if not all(field in data for field in required_fields):
#         return jsonify({"error": "数据格式不正确"}), 400
        
#     if not data['timesteps'] or not isinstance(data['timesteps'], list):
#         return jsonify({"error": "timesteps 必须是非空列表"}), 400
    
#     try:
#         # 读取现有的 traces
#         with open(TRACE_FILE, 'r', encoding='utf-8') as f:
#             traces = json.load(f)
#     except json.JSONDecodeError:
#         traces = []
    
#     # 查找是否存在相同的 question_idx
#     found = False
#     for trace in traces:
#         if trace['question_idx'] == data['question_idx']:
#             # 如果找到相同的 question_idx，则添加新的 timesteps
#             trace['timesteps'].extend(data['timesteps'])
#             found = True
#             break
    
#     # 如果没有找到相同的 question_idx，则添加新记录
#     if not found:
#         new_trace = {
#             'question_idx': data['question_idx'],
#             'timesteps': data['timesteps']
#         }
#         traces.append(new_trace)
    
#     try:
#         # 保存到文件
#         with open(TRACE_FILE, 'w', encoding='utf-8') as f:
#             json.dump(traces, f, ensure_ascii=False, indent=4)
#     except Exception as e:
#         print(f"保存轨迹时出错: {e}")
#         return jsonify({"error": "保存轨迹失败"}), 500
    
#     return jsonify({"status": "success"}), 200

# @app.route('/log', methods=['POST'])
# def log_interaction():
#     data = request.get_json()
#     if not data:
#         print("No data received")
#         return jsonify({"error": "No data provided"}), 400
    
#     print(f"Received data: {data}")  # 调试信息

#     # 读取现有日志
#     try:
#         with open(LOG_FILE, 'r', encoding='utf-8') as f:
#             logs = json.load(f)
#     except Exception as e:
#         print(f"Error reading log file: {e}")
#         logs = []

#     # 添加新日志
#     logs.append(data)
    
#     # 保存日志
#     try:
#         with open(LOG_FILE, 'w', encoding='utf-8') as f:
#             json.dump(logs, f, ensure_ascii=False, indent=4)
#         print(f"Log saved: {data}")  # 调试信息
#     except Exception as e:
#         print(f"Error writing to log file: {e}")
#         return jsonify({"error": "Failed to write log"}), 500

#     return jsonify({"status": "success"}), 200



@app.route('/<city>/trace', methods=['POST'])
def city_save_trace(city):
    # 如果城市不在白名单，返回404
    if city not in ALLOWED_CITIES:
        return jsonify({"error": "城市不被支持"}), 404

    data = request.get_json()
    if not data:
        return jsonify({"error": "没有提供数据"}), 400
    
    # 添加数据验证
    required_fields = ['question_idx', 'timesteps']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "数据格式不正确"}), 400
    
    if not data['timesteps'] or not isinstance(data['timesteps'], list):
        return jsonify({"error": "timesteps 必须是非空列表"}), 400

    # ★ 用 city 拼出专属 trace 文件名，例如 BeiJing_1_trace.json
    city_trace_file = f"{city}_trace.json"

    # 如果不存在就先创建一个空的
    if not os.path.exists(city_trace_file):
        with open(city_trace_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)

    # 读取现有的 traces
    try:
        with open(city_trace_file, 'r', encoding='utf-8') as f:
            traces = json.load(f)
    except json.JSONDecodeError:
        traces = []

    # 查找是否存在相同的 question_idx
    found = False
    for trace in traces:
        if trace['question_idx'] == data['question_idx']:
            # 如果找到相同的 question_idx，则添加新的 timesteps
            trace['timesteps'].extend(data['timesteps'])
            found = True
            break
    
    # 如果没有找到相同的 question_idx，则添加新记录
    if not found:
        new_trace = {
            'question_idx': data['question_idx'],
            'timesteps': data['timesteps']
        }
        traces.append(new_trace)
    
    # 保存到文件
    try:
        with open(city_trace_file, 'w', encoding='utf-8') as f:
            json.dump(traces, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"保存轨迹时出错: {e}")
        return jsonify({"error": "保存轨迹失败"}), 500
    
    return jsonify({"status": "success"}), 200


@app.route('/<city>/log', methods=['POST'])
def city_log_interaction(city):
    # 如果城市不在白名单，返回404
    if city not in ALLOWED_CITIES:
        return jsonify({"error": "城市不被支持"}), 404

    data = request.get_json()
    if not data:
        print("No data received")
        return jsonify({"error": "No data provided"}), 400
    
    print(f"Received data: {data}")  # 调试信息

    # ★ 用 city 拼出专属 log 文件名，例如 BeiJing_1_interaction_logs.json
    city_log_file = f"{city}_interaction_logs.json"

    # 如果不存在就先创建一个空的
    if not os.path.exists(city_log_file):
        with open(city_log_file, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)

    # 读取现有日志
    try:
        with open(city_log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    except Exception as e:
        print(f"Error reading log file: {e}")
        logs = []

    # 添加新日志
    logs.append(data)
    
    # 保存日志
    try:
        with open(city_log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)
        print(f"Log saved: {data}")  # 调试信息
    except Exception as e:
        print(f"Error writing to log file: {e}")
        return jsonify({"error": "Failed to write log"}), 500

    return jsonify({"status": "success"}), 200




# 处理静态文件（如CSS、JS等）
@app.route('/static/<path:filename>')
def serve_static(filename):
    static_dir = os.path.join(app.root_path, 'static')
    file_path = os.path.join(static_dir, filename)
    if not os.path.exists(file_path):
        print(f"Static file not found: {file_path}")
        return jsonify({"error": "文件不存在"}), 404
    return send_file(file_path)




if __name__ == '__main__':
    app.run(port=5000, debug=True)
