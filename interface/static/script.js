document.getElementById('startCourseBtn').addEventListener('click', function() {
    document.getElementById('videoContainer').classList.remove('hidden');
    startWebcam();
});

function startWebcam() {
    const webcamElement = document.getElementById('webcam');
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function(stream) {
            webcamElement.srcObject = stream;
            const socket = io.connect('http://localhost:5000');

            socket.on('connect', () => {
                console.log('Connected to server');
                socket.on('server_message', (message) => {
                    console.log(message.data);
                });
            });

            socket.on('regular_status', (message) => {
                console.log(message.status);
            });

            socket.on('attention_status', (message) => {
                console.log(message.status);
                if (message.status === "STUDENT YOU ARE NOT PAYING ATTENTION") {
                    showAlert(message.status, socket);
                }
            });

            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');

            setInterval(() => {
                canvas.width = webcamElement.videoWidth;
                canvas.height = webcamElement.videoHeight;
                context.drawImage(webcamElement, 0, 0, canvas.width, canvas.height);
                const imageData = canvas.toDataURL('image/jpeg');
                socket.emit('image', imageData);
            }, 1000);
        })
        .catch(function(err) {
            console.error('Error accessing webcam: ', err);
        });
}

function showAlert(message, socket) {
    const alertDiv = document.createElement('div');
    alertDiv.innerHTML = `<h1 style="color: red;">${message}</h1><button id="acknowledgeBtn">OK</button>`;
    alertDiv.style.position = 'fixed';
    alertDiv.style.top = '50%';
    alertDiv.style.left = '50%';
    alertDiv.style.transform = 'translate(-50%, -50%)';
    alertDiv.style.backgroundColor = 'white';
    alertDiv.style.padding = '20px';
    alertDiv.style.boxShadow = '0 0 10px rgba(0, 0, 0, 0.5)';
    alertDiv.style.zIndex = '1000';
    document.body.appendChild(alertDiv);

    document.getElementById('acknowledgeBtn').addEventListener('click', function() {
        document.body.removeChild(alertDiv);
        socket.emit('acknowledge_alert');
    });
}
