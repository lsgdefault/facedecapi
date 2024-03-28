from flask import Flask, jsonify, request
import cv2
import face_recognition
import numpy as np
import os

app = Flask(__name__)

# Load reference images from a directory
reference_images_folder = 'reference_images'
reference_encodings = []

# Load and encode all images from the reference images folder
for filename in os.listdir(reference_images_folder):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        image_path = os.path.join(reference_images_folder, filename)
        reference_image = face_recognition.load_image_file(image_path)
        reference_encoding = face_recognition.face_encodings(reference_image)[0]
        reference_encodings.append({'image_name': filename, 'encoding': reference_encoding})

# Endpoint for processing images
@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Read the image from the request
        image = request.files['image'].read()
        nparr = np.frombuffer(image, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and encodings in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        results = []

        # If no faces are found, return an empty result
        if len(face_encodings) == 0:
            return jsonify([])

        # Compare each face encoding with the reference encodings
        for face_encoding in face_encodings:
            # Compare the current face encoding with the reference encodings
            matches = []
            for ref_encoding in reference_encodings:
                match = face_recognition.compare_faces([ref_encoding['encoding']], face_encoding)
                if True in match:
                    matches.append(ref_encoding['image_name'])

            if matches:
                results.append({'match': True, 'image_names': matches})
            else:
                results.append({'match': False, 'image_names': ['unknown']})

        return jsonify(results)

    except Exception as e:
        # Handle errors gracefully
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
