import cv2
import numpy as np

# -------------------------------------------------------
# contadorCarros.py
# Conta carros que cruzam uma linha virtual na rua.
# -------------------------------------------------------

# Abrir o vídeo — coloque o arquivo na mesma pasta ou use o caminho completo
video = cv2.VideoCapture('video.mp4')

if not video.isOpened():
    print("ERRO: não foi possível abrir o vídeo. Verifique o nome do arquivo.")
    exit()

# Subtrator de fundo — aprende o fundo e isola o que se move
subtrator = cv2.createBackgroundSubtractorMOG2(
    history=100,       # quantos frames para aprender o fundo
    varThreshold=40,   # sensibilidade (aumente se tiver muitos falsos positivos)
    detectShadows=True # detecta sombras para removê-las
)

# ── Configurações — AJUSTE CONFORME SEU VÍDEO ──────────────
LINHA_Y  = 300   # posição Y da linha de contagem (em pixels)
MIN_AREA = 1200  # área mínima do blob para ser considerado um carro
# ───────────────────────────────────────────────────────────

contador      = 0
carros_rastreados = {}  # { id: (cx, cy, lado) }
proximo_id    = 0

COR_LINHA = (0, 255, 255)   # amarelo
COR_CARRO = (0, 200, 0)     # verde
COR_TEXTO = (255, 255, 255) # branco
COR_PONTO = (255, 0, 255)   # magenta


def get_centro(x, y, w, h):
    return x + w // 2, y + h // 2


while True:
    ret, frame = video.read()

    if not ret:
        print("Fim do vídeo.")
        break

    frame = cv2.resize(frame, (800, 600))
    altura, largura = frame.shape[:2]

    # 1. Subtração de fundo
    mascara = subtrator.apply(frame)

    # Remove sombras (pixels cinza vão a preto)
    _, mascara = cv2.threshold(mascara, 200, 255, cv2.THRESH_BINARY)

    # 2. Morfologia — remove ruído e une partes do mesmo carro
    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.erode(mascara,  kernel, iterations=1)
    mascara = cv2.dilate(mascara, kernel, iterations=3)

    # 3. Detecta contornos (cada blob = objeto em movimento)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centros_frame = []

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue  # muito pequeno → não é carro

        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = get_centro(x, y, w, h)
        centros_frame.append((cx, cy, x, y, w, h))

        cv2.rectangle(frame, (x, y), (x + w, y + h), COR_CARRO, 2)
        cv2.circle(frame, (cx, cy), 5, COR_PONTO, -1)

    # 4. Rastreamento por proximidade + lógica de cruzamento
    novos_rastreados = {}

    for (cx, cy, x, y, w, h) in centros_frame:
        melhor_id    = None
        menor_dist   = 80  # px — distância máxima para associar ao mesmo carro

        for cid, (px, py, _) in carros_rastreados.items():
            dist = np.hypot(cx - px, cy - py)
            if dist < menor_dist:
                menor_dist = dist
                melhor_id  = cid

        if melhor_id is None:
            melhor_id   = proximo_id
            proximo_id += 1

        lado_atual = "acima" if cy < LINHA_Y else "abaixo"

        if melhor_id in carros_rastreados:
            _, _, lado_anterior = carros_rastreados[melhor_id]
            if lado_anterior == "acima" and lado_atual == "abaixo":
                contador += 1
                print(f"Carro #{contador} contado!")
            # Para contar também no sentido contrário, descomente:
            # elif lado_anterior == "abaixo" and lado_atual == "acima":
            #     contador += 1

        novos_rastreados[melhor_id] = (cx, cy, lado_atual)

    carros_rastreados = novos_rastreados

    # 5. Desenha a interface
    cv2.line(frame, (0, LINHA_Y), (largura, LINHA_Y), COR_LINHA, 2)
    cv2.putText(frame, "LINHA DE CONTAGEM", (10, LINHA_Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_LINHA, 1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (290, 68), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, f"Carros: {contador}",
                (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.3, COR_TEXTO, 2)

    cv2.imshow("Contador de Carros", frame)
    cv2.imshow("Mascara (movimento)", cv2.resize(mascara, (400, 300)))

    if cv2.waitKey(30) == 27:  # ESC para sair
        break

video.release()
cv2.destroyAllWindows()
print(f"\nTotal final: {contador} carros contados.")