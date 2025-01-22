from flask import Flask, request, jsonify
from ultralytics import YOLO
# Import the SDK and the client module
from label_studio_sdk.client import LabelStudio
from PIL import Image
import requests
from io import BytesIO
import uuid

app = Flask(__name__)

#Configuracion URL de Label Studio y token de autenticación
label_studio_url = "http://192.168.0.15:8080"
api_token = "3a86d046218d1e73c42a5f0f5e0424cbd03f9137"
model_path = "ppe_yolov8.pt"

#Definir los encabezados con el token de autenticacion
headers = {
    "Authorization" : f"Token {api_token}",
    "Content-Type": "application/json"
}

#Cliente de label Studio
ls_client = LabelStudio(base_url = label_studio_url, api_key = api_token)

#Carga de modelo
model = YOLO(model_path)

def get_image(url_relative):
    #Construir la URL completa con host y token de cabeceras
    complete_url = f"{label_studio_url}{url_relative}"
    print(f"La ruta de la imagen es: {complete_url}")
    try:
        response = requests.get(complete_url, headers = headers)
        # Check for authorization error
        if response.status_code == 401:
            print("Authorization failed. Check your API token or Label Studio URL.")
            return None
        elif response.status_code != 200:
            print(f"Failed to fetch image: {response.status_code}")
            return None

        # Load and return the image
        return Image.open(BytesIO(response.content))
    except requests.exceptions.HTTPError as err:
        print(f"Error al obtener la imagen: {err}")

def predict(task):
    print("Iniciando predicción...")
    predictions = []
    try:
        # for task in tasks:
            # Cargar la imagen del task
            image_relative = task["data"]["image"]

            print(f"El argumento que pasa es: {image_relative}")

            image_url = get_image(image_relative)

            try:
                results = model(image_url)

                # Si results es una lista (por ejemplo, si hay un batch de imágenes)
                if isinstance(results, list):
                    results = results[0]  # Accedemos al primer elemento de la lista

                # Asegurarnos de que results tiene los atributos esperados
                if hasattr(results, 'boxes') and results.boxes is not None:
                    boxes = results.boxes.xyxy  # Esto contiene las coordenadas de las cajas
                    print(f"Cajas: {boxes}")

                    if boxes is not None and len(boxes) > 0:
                        for i, box in enumerate(boxes.tolist()):
                            # Desempaquetar solo 4 valores para las coordenadas de las cajas
                            x_min, y_min, x_max, y_max = box

                            prediction_id = str(uuid.uuid4())

                            # Valor predeterminado para label
                            label = "Desconocido"

                            # Acceder a las probabilidades (confianza) si están disponibles
                            if hasattr(results, 'probs') and results.probs is not None:
                                confidence = results.probs[i].item()  # Confianza para esta caja

                            # Obtener el nombre de la clase si está disponible
                            if hasattr(results, 'names'):
                                label = results.names[int(results.boxes.cls[i].item())]  # Usamos el índice de clase

                            # Solo agregamos las predicciones de EPIs
                            if label in ["Gloves", "Hardhat", "Glasses", "Head", "Person", "Safety-Vest"]:
                                predictions.append(
                                        {
                                            "original_width": image_url.width,   # Ancho original de la imagen
                                            "original_height": image_url.height, # Alto original de la imagen
                                            "image_rotation": 0,             # Si hay rotación en la imagen, la puedes ajustar aquí
                                            "from_name": "label",          # Nombre del campo de predicción
                                            "to_name": "image",             # Nombre del campo de la imagen
                                            "type": "rectanglelabels",      # Tipo de anotación
                                            "id": prediction_id,  # ID único
                                            "value": {
                                                "x": (x_min / image_url.width) * 100,    # Coordenada X en porcentaje
                                                "y": (y_min / image_url.height) * 100,   # Coordenada Y en porcentaje
                                                "width": ((x_max - x_min) / image_url.width) * 100,  # Ancho del rectángulo en porcentaje
                                                "height": ((y_max - y_min) / image_url.height) * 100, # Alto del rectángulo en porcentaje
                                                "rotation": 0,  # Rotación de la predicción (opcional)
                                                "rectanglelabels": [label]  # Etiqueta predicha
                                            }

                                        }
                                    )
                    else:
                        print("No se encontraron cajas en la imagen.")
                else:
                    print(f"Resultados vacíos para {image_url}. No se encontraron cajas.")                
            except Exception as e:
                print(f"Error al realizar predicción en la imagen {image_url}: {e}")

            print(f"Predicción completada y su tipo es: {type(predictions)}")
            return predictions
        
    except Exception as e:
            print(f"Error al realizar predicción en la imagen {image_url}: {e}")

    return predictions

#Funciones de union con Label Studio
# Rutas de la API para la verificación de estado y configuración inicial
# Ruta para el estado del backend ML (healthy check)
@app.route('/health', methods=['GET'])
def health_check():
    response = {
        'status': 'UP',
        'model': 'YOLOv8',
        'version': '1.0'
    }
    return jsonify(response), 200

# Ruta para la configuración del backend ML
@app.route('/setup', methods=['GET', 'POST'])
def setup():
        try:
            # Si es un POST y se envían datos en el cuerpo
            if request.method == 'POST':
                data = request.json  # Obtener los datos del cuerpo de la solicitud
                response = {
                    'model': 'YOLOv8',
                    'status': 'ok',
                    'version': '1.0',
                    'setup_data_received': data  # Opcional: los datos recibidos
                }
            else:
                # Si es un GET
                response = {
                    'model': 'YOLOv8',
                    'status': 'ok',
                    'version': '1.0'
                }
            return jsonify(response), 200
        except Exception as e:
            # Si ocurre algún error, devuelve un mensaje de error en el formato correcto
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
# Ruta para la predicción
@app.route("/predict", methods = ["POST"])
def predict_route():
    print("Entro aqui")
    try:
        tasks = request.json.get('tasks', None)  # Usar get_json() en lugar de request.json, por seguridad

        for task in tasks:
            print(f"Recibimos estos datos: {len(tasks)}")
            print(f"Recibimos estos datos: {tasks}")
            task_id = task['id'] 
            print(f"El id del task es: {task_id}")
            print("Vamos a realizar las predicciones")
            #Hora de hacer las predicciones
            _predictions = predict(task)

            if not _predictions:
                return jsonify({"error": "No se generaron predicciones"}), 500
            
            print(f"Actualizaremos con estas predicciones: {_predictions} y el tipo de variable es: {type(_predictions)}")
            print("Vamos a actualizar las predicciones en LabelStudio")
            # # #Actualizar las predicciones en label Studio

            try:
                existing_predictions = ls_client.predictions.list(task=task_id)
                print(f"La prediccion existente es: {existing_predictions}")
                if existing_predictions:
                    # Si existe una predicción, actualizarla
                    prediction_id = existing_predictions[0].id  # Usar la primera predicción
                    print(f"Actualizando predicción existente con ID: {prediction_id}")
                    
                    ls_client.predictions.update(
                        id=prediction_id,
                        result=_predictions,
                        score=0.95,
                        model_version="YOLOV8",
                    )
                else:
                    # Si no existe, crear una nueva
                    print(f"Creando nueva predicción para la tarea {task_id}")
                    ls_client.predictions.create(
                        task = task_id,
                        result= _predictions,    
                        score=0.95,
                        model_version="YOLOV8",
                    )   
            
            except Exception as e:
                print(f"El error es: {e}")
        
        # print(f"Mandamos estas predicciones: {response}")
        return jsonify({'status': 'Todo correcto!', 'message': str(e)}), 200
    
    except Exception as e:
        return jsonify({"error" : str(e)}), 500
    
#Ruta para cuando se haya iniciado correctamente
@app.route('/', methods=['GET'])
def index():
    return "Bienvenido al YOLOv8 Backend!"

#Ruta para cuando recibe una tarea
@app.route("/", methods=["POST"])
def handle_post():
    return "Post recibido", 200

#Ruta para cuando tiene webhook
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json  # Obtener los datos de la solicitud POST
    print(data)  # Aquí procesas la predicción o los datos del webhook
    return {"status": "success", "message": "Predicción recibida"}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9090, debug=True)
