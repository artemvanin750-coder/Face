<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Распознавание лиц (Пол, Возраст, Эмоции)</title>
    
    <!-- Шрифт для лучшей читаемости -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">

    <style>
        :root {
            --bg: #ffffff;
            --text: #2d3748;
            --muted: #718096;
            --primary: #4a5568;
            --accent: #4caf50;
            --border: #e2e8f0;
        }

        * {
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            display: flex;
            justify-content: center;
            align-items: flex-start;
            min-height: 100vh;
            padding: 20px;
            margin: 0;
        }

        .container {
            width: 100%;
            max-width: 600px;
            background: #fafafa;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        }

        h1 {
            font-size: 24px;
            font-weight: 600;
            margin: 0 0 20px 0;
            text-align: center;
            color: var(--primary);
        }

        .upload-area {
            text-align: center;
            margin-bottom: 20px;
        }

        #file-input {
            display: none;
        }

        .upload-label {
            display: inline-block;
            padding: 14px 24px;
            background-color: var(--primary);
            color: #fff;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 500;
            cursor: pointer;
            transition: background-color 0.2s;
            user-select: none;
        }

        .upload-label:hover {
            background-color: #414a5c;
        }

        .status {
            text-align: center;
            margin-top: 12px;
            font-size: 14px;
            color: var(--muted);
            min-height: 20px;
        }

        .disclaimer {
            text-align: center;
            font-size: 13px;
            color: var(--muted);
            margin-top: 24px;
            padding-top: 20px;
            border-top: 1px dashed var(--border);
            line-height: 1.4;
        }

        .result-container {
            position: relative;
            margin-top: 30px;
            text-align: center;
            display: none; /* Скрыт до загрузки фото */
        }

        #result-image {
            width: 100%;
            height: auto;
            border-radius: 8px;
            max-width: 600px;
            background-color: #000;
        }

        #overlay-canvas {
            position: absolute;
            top: 0;
            left: 50%;
            transform: translateX(-50%);
            border-radius: 8px;
            pointer-events: none;
            max-width: 600px;
        }

        .face-info {
            position: absolute;
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 6px 10px;
            font-size: 12px;
            font-weight: 500;
            color: var(--text);
            pointer-events: none;
            user-select: none;
            white-space: nowrap;
            line-height: 1.3;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
</head>
<body>

    <div class="container">
        <h1>Анализ лиц на фотографии</h1>

        <div class="upload-area">
            <input type="file" id="file-input" accept="image/*" capture="environment">
            <label for="file-input" class="upload-label">Выбрать фотографию</label>
            <div id="status-text" class="status">Загрузка моделей...</div>
        </div>

        <div class="result-container" id="result-area">
            <img id="result-image" src="" alt="Загруженное фото">
            <canvas id="overlay-canvas"></canvas>
        </div>

        <div class="disclaimer">
            Ваши фото обрабатываются только в вашем браузере и никуда не отправляются. Изображение не сохраняется.
        </div>
    </div>

    <!-- Библиотеки: TensorFlow.js и Face-api.js -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0/dist/tf.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js"></script>

    <script>
        // --- Конфигурация ---
        const MODEL_CDN = 'https://cdn.jsdelivr.net/npm/face-api@0.22.2/weights/';
        const STATUS_TEXT = document.getElementById('status-text');
        const FILE_INPUT = document.getElementById('file-input');
        const RESULT_AREA = document.getElementById('result-area');
        const RESULT_IMG = document.getElementById('result-image');
        const OVERLAY_CANVAS = document.getElementById('overlay-canvas');

        // --- 1. Загрузка моделей (асинхронно при старте) ---
        async function loadModels() {
            try {
                await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_CDN);
                await faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_CDN);
                await faceapi.nets.ageGenderNet.loadFromUri(MODEL_CDN);
                await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_CDN);
                
                STATUS_TEXT.textContent = 'Модели загружены. Выберите фото.';
            } catch (error) {
                STATUS_TEXT.textContent = 'Ошибка загрузки моделей. Проверьте интернет.';
                console.error(error);
            }
        }

        // --- 2. Отрисовка результатов на Canvas ---
        function drawResults(detections, canvas, displaySize) {
            const ctx = canvas.getContext('2d');
            
            // Очищаем предыдущий кадр
            canvas.width = displaySize.width;
            canvas.height = displaySize.height;
            
            // Рисуем рамку и подпись для каждого лица
            detections.forEach(detection => {
                const box = detection.detection.box;
                const { gender, age, genderProbability, expressions } = detection;

                // Определяем доминирующую эмоцию
                let dominantEmotion = 'Нейтрально';
                let maxScore = 0;
                if (expressions) {
                    Object.keys(expressions).forEach(emotion => {
                        if (expressions[emotion] > maxScore) {
                            maxScore = expressions[emotion];
                            dominantEmotion = emotion.charAt(0).toUpperCase() + emotion.slice(1);
                        }
                    });
                }

                // Рисуем рамку
                ctx.strokeStyle = '#4caf50';
                ctx.lineWidth = 3;
                ctx.strokeRect(box.x, box.y, box.width, box.height);

                // Формируем текст подписи
                const genderText = gender === 'male' ? 'Мужской' : 'Женский';
                const label = `Пол: ${genderText}, Возраст: ${Math.round(age)}, Эмоция: ${dominantEmotion}`;
                
                // Рисуем фон для текста
                ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
                const textWidth = ctx.measureText(label).width + 12;
                ctx.fillRect(box.x, box.y - 30, textWidth, 26);

                // Рисуем текст
                ctx.fillStyle = '#2d3748';
                ctx.font = '13px Inter, sans-serif';
                ctx.textBaseline = 'middle';
                ctx.fillText(label, box.x + 6, box.y - 17);
            });
        }

        // --- 3. Основная логика обработки файла ---
        async function handleImageUpload(event) {
            const file = event.target.files[0];
            if (!file) return;

            STATUS_TEXT.textContent = 'Распознаю лица...';
            RESULT_AREA.style.display = 'none'; // Скрываем на время обработки

            // Создаем URL для отображения картинки
            const imgUrl = URL.createObjectURL(file);
            RESULT_IMG.src = imgUrl;

            // Ждем, пока картинка загрузится в тег img
            await new Promise(resolve => { RESULT_IMG.onload = resolve; });

            // Создаем временный canvas для face-api.js (она работает с Canvas/Image)
            const displaySize = { width: RESULT_IMG.width, height: RESULT_IMG.height };
            const inputCanvas = document.createElement('canvas');
            inputCanvas.width = displaySize.width;
            inputCanvas.height = displaySize.height;
            const inputCtx = inputCanvas.getContext('2d');
            inputCtx.drawImage(RESULT_IMG, 0, 0, displaySize.width, displaySize.height);

            // Выполняем инференс (детекция + возраст/пол/эмоции)
            const detections = await faceapi
                .detectAllFaces(inputCanvas, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }))
                .withFaceLandmarks(true)
                .withAgeAndGender()
                .withFaceExpressions();

            // Масштабируем canvas-оверлей под размер изображения
            faceapi.matchDimensions(OVERLAY_CANVAS, displaySize);
            drawResults(detections, OVERLAY_CANVAS, displaySize);

            // Показываем результат
            RESULT_AREA.style.display = 'block';
            STATUS_TEXT.textContent = `Найдено лиц: ${detections.length}`;
        }

        // --- Инициализация ---
        (async () => {
            await loadModels();
            FILE_INPUT.addEventListener('change', handleImageUpload);
        })();
    </script>
</body>
</html>
