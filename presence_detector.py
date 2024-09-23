import face_recognition
import cv2
import streamlit as st
import numpy as np
import os
import tempfile
from datetime import datetime
from mtcnn import MTCNN

# Configurações de desempenho (valores padrão)
DEFAULT_FRAME_RESIZE_FACTOR = 0.5  # Fator de redimensionamento para acelerar a detecção
DEFAULT_BATCH_SIZE = 10             # Número de quadros a processar antes de atualizar a barra de progresso
DEFAULT_FRAME_SKIP = 1              # Número de quadros a pular (0 = nenhum, 1 = processar todos, 2 = pular 1, etc.)
DEFAULT_TOLERANCE = 0.5             # Tolerância para reconhecimento facial (reduzido para maior precisão)

# Inicialização das variáveis de sessão para armazenar o estado
if 'processed_video_name' not in st.session_state:
    st.session_state['processed_video_name'] = None
if 'processed_video_bytes' not in st.session_state:
    st.session_state['processed_video_bytes'] = None
if 'output_filename' not in st.session_state:
    st.session_state['output_filename'] = None

# Decorador para carregar o detector MTCNN uma única vez
@st.cache_resource(show_spinner=False)
def initialize_detector(min_face_size=40, steps_threshold=[0.6, 0.7, 0.7]):
    return MTCNN(min_face_size=min_face_size, steps_threshold=steps_threshold)

# Decorador de cache para carregar as faces conhecidas
@st.cache_data(show_spinner=False)
def load_known_faces(known_faces_dir):
    known_encodings = []
    known_names = []
    
    if not os.path.exists(known_faces_dir):
        st.error(f"O diretório '{known_faces_dir}' não existe. Por favor, verifique o caminho.")
        return np.array(known_encodings), known_names

    # Percorre todas as imagens no diretório de faces conhecidas
    for file in os.listdir(known_faces_dir):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(known_faces_dir, file)
            try:
                img = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(img)
                if encodings:
                    # Adiciona todas as codificações encontradas na imagem
                    for encoding in encodings:
                        known_encodings.append(encoding)
                        known_names.append(os.path.splitext(file)[0])
                else:
                    st.warning(f"Nenhuma face encontrada na imagem {file}.")
            except Exception as e:
                st.warning(f"Erro ao processar {file}: {e}")
    st.write(f"Total de rostos conhecidos carregados: {len(known_encodings)}")
    return np.array(known_encodings), known_names

def detect_faces_mtcnn(detector, frame, resize_factor):
    # Redimensiona o quadro para acelerar a detecção
    if resize_factor != 1.0:
        small_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
    else:
        small_frame = frame.copy()
    
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Detecta as faces usando MTCNN
    detections = detector.detect_faces(rgb_small_frame)
    
    face_locations = []
    for det in detections:
        x, y, width, height = det['box']
        # Garante que as coordenadas não fiquem negativas
        x, y = max(0, x), max(0, y)
        # Reverte o redimensionamento
        top = int(y / resize_factor)
        right = int((x + width) / resize_factor)
        bottom = int((y + height) / resize_factor)
        left = int(x / resize_factor)
        face_locations.append((top, right, bottom, left))
    
    return face_locations

def process_frame(frame, detector, known_encodings, known_names, tolerance, resize_factor):
    # Detecta as localizações das faces usando MTCNN
    face_locations = detect_faces_mtcnn(detector, frame, resize_factor)
    
    # Codifica as faces detectadas
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    face_names = []
    if len(face_encodings) > 0 and len(known_encodings) > 0:
        # Itera sobre cada face_encoding detectada
        for face_encoding in face_encodings:
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(distances) > 0:
                best_match_index = np.argmin(distances)
                if distances[best_match_index] < tolerance:
                    name = known_names[best_match_index]
                else:
                    name = "Desconhecido"
            else:
                name = "Desconhecido"
            face_names.append(name)
    else:
        face_names = ["Desconhecido"] * len(face_encodings)
    
    # Desenha as caixas e os nomes das faces
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if name != "Desconhecido":
            box_color = (0, 255, 0)  # Verde para conhecido
            label_color = (0, 255, 0)  # Verde para o fundo da etiqueta
        else:
            box_color = (0, 0, 255)  # Vermelho para desconhecido
            label_color = (0, 0, 255)  # Vermelho para o fundo da etiqueta
        
        # Desenha a caixa ao redor da face
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
        
        # Define as dimensões da etiqueta
        label_height = 30
        label_y_min = max(bottom - label_height, 0)
        
        # Desenha a etiqueta com o nome
        cv2.rectangle(frame, (left, label_y_min), (right, bottom), label_color, cv2.FILLED)
        
        # Configurações da fonte
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        
        # Calcula a largura e altura do texto
        text_size, _ = cv2.getTextSize(name, font, font_scale, font_thickness)
        text_width, text_height = text_size
        
        # Calcula a posição do texto para centralizá-lo na etiqueta
        text_x = left + 6
        text_y = bottom - 6
        
        # Ajusta a cor do texto para contraste
        text_color = (255, 255, 255)  # Branco
        
        # Desenha o texto na etiqueta
        cv2.putText(frame, name, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return frame, len(face_locations)

def process_video(video_path, known_encodings, known_names, output_path, tolerance, resize_factor, batch_size, frame_skip):
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        st.error(f"Não foi possível abrir o vídeo: {video_path}")
        return False
    
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Verificar se os parâmetros estão corretos
    if fps == 0:
        st.error("FPS do vídeo é 0.")
        video_capture.release()
        return False
    if frame_width == 0 or frame_height == 0:
        st.error("Resolução do vídeo inválida.")
        video_capture.release()
        return False
    
    # Definir o codec para 'mp4v' para contêiner MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'avc1' ou 'H264' pode não ser suportado em todas as instalações
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        st.error(f"Não foi possível abrir o VideoWriter com o caminho: {output_path}")
        video_capture.release()
        return False
    
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    
    frame_number = 0
    processed_frames = 0
    detector = initialize_detector()
    
    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Pular quadros conforme FRAME_SKIP
            if frame_skip > 0 and frame_number % (frame_skip + 1) != 0:
                out.write(frame)  # Escreve o quadro original sem processamento
                frame_number += 1
                continue
            
            processed_frame, faces_detected = process_frame(
                frame, detector, known_encodings, known_names, tolerance, resize_factor
            )
            out.write(processed_frame)
            frame_number += 1
            processed_frames += 1
            
            # Atualiza a barra de progresso a cada BATCH_SIZE quadros processados
            if processed_frames % batch_size == 0 or frame_number == total_frames:
                progress = frame_number / total_frames
                progress_bar.progress(min(progress, 1.0))
    except Exception as e:
        st.error(f"Erro durante o processamento do vídeo: {e}")
        video_capture.release()
        out.release()
        return False
    finally:
        video_capture.release()
        out.release()
        progress_bar.empty()
    
    return True

# Interface com Streamlit
def main():
    st.set_page_config(page_title="Sistema de Detecção e Reconhecimento Facial Otimizado", layout="wide")
    
    # Título e Subtítulo
    st.title("Sistema de Detecção e Reconhecimento Facial Otimizado")
    st.markdown("""
    **Bem-vindo ao nosso sistema avançado de detecção e reconhecimento facial!**
    
    Esta ferramenta foi desenvolvida para **identificar e reconhecer rostos** em vídeos de forma rápida e precisa. Utilizando as mais recentes tecnologias de visão computacional e aprendizado de máquina, o sistema oferece uma experiência eficiente para diversas aplicações, como segurança, análise de público e muito mais.
    """)
    
    # Descrição das Funcionalidades
    st.markdown("""
    ### **Funcionalidades Principais**
    - **Detecção de Rostos:** Utiliza a biblioteca `MTCNN` para identificar a localização de rostos em cada quadro do vídeo.
    - **Reconhecimento Facial:** Com base nas codificações geradas pelo `face_recognition`, o sistema compara e identifica rostos conhecidos.
    - **Processamento Otimizado:** Configurações ajustáveis para equilibrar entre velocidade e precisão, incluindo redimensionamento de quadros e pular quadros para acelerar o processamento.
    - **Interface Intuitiva:** Desenvolvido com `Streamlit`, oferecendo uma interface amigável para upload e visualização de vídeos processados.
    - **Download Fácil:** Após o processamento, o vídeo com as identificações é exibido e pode ser baixado diretamente pela interface.
    """)
    
    # Instruções de Uso
    st.markdown("""
    ### **Como Utilizar**
    1. **Carregar o Vídeo:**
       - Clique no botão de upload e selecione um vídeo nos formatos `mp4`, `avi` ou `mov`.
    2. **Ajustar as Configurações:**
       - **Fator de Redimensionamento de Quadro:** Reduza a resolução dos quadros para acelerar o processamento. Valores menores aumentam a velocidade, mas podem reduzir a precisão.
       - **Tamanho do Lote:** Defina o número de quadros a serem processados antes de atualizar a barra de progresso.
       - **Pular Quadros:** Especifique quantos quadros pular entre os processados. Por exemplo, `1` significa processar um quadro a cada dois.
       - **Tolerância de Reconhecimento:** Ajuste o quão estrita a correspondência deve ser para considerar uma face como conhecida. Valores menores aumentam a precisão.
    3. **Processar o Vídeo:**
       - Após ajustar as configurações, aguarde enquanto o sistema processa o vídeo. A barra de progresso indicará o andamento.
    4. **Visualizar e Baixar:**
       - Uma vez concluído, o vídeo processado será exibido na interface e você poderá baixá-lo diretamente.
    """)
    
    # Separador visual
    st.markdown("---")
    
    # Configurações no Sidebar
    st.sidebar.header("Configurações")
    frame_resize_factor = st.sidebar.slider(
        "Fator de Redimensionamento de Quadro",
        min_value=0.1,
        max_value=1.0,
        value=DEFAULT_FRAME_RESIZE_FACTOR,
        step=0.1,
        help="Reduz a resolução do quadro para acelerar o processamento. Valores menores aumentam a velocidade, mas podem reduzir a precisão."
    )
    batch_size = st.sidebar.slider(
        "Tamanho do Lote",
        min_value=1,
        max_value=50,
        value=DEFAULT_BATCH_SIZE,
        step=1,
        help="Número de quadros a processar antes de atualizar a barra de progresso."
    )
    frame_skip = st.sidebar.slider(
        "Pular Quadros",
        min_value=0,
        max_value=10,
        value=DEFAULT_FRAME_SKIP,
        step=1,
        help="Número de quadros a pular entre os processados. 0 = processar todos, 1 = processar um em cada dois, etc."
    )
    tolerance = st.sidebar.slider(
        "Tolerância de Reconhecimento",
        min_value=0.3,
        max_value=0.6,
        value=DEFAULT_TOLERANCE,
        step=0.01,
        help="Valor de tolerância para o reconhecimento facial. Valores menores aumentam a precisão, mas podem reduzir a taxa de correspondência."
    )
    
    uploaded_file = st.file_uploader("Carregar Vídeo", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # Verifica se um novo arquivo foi carregado ou se os parâmetros foram alterados
        new_file = uploaded_file.name != st.session_state.get('processed_video_name', None)
        params_changed = (
            st.session_state.get('frame_resize_factor', DEFAULT_FRAME_RESIZE_FACTOR) != frame_resize_factor or
            st.session_state.get('batch_size', DEFAULT_BATCH_SIZE) != batch_size or
            st.session_state.get('frame_skip', DEFAULT_FRAME_SKIP) != frame_skip or
            st.session_state.get('tolerance', DEFAULT_TOLERANCE) != tolerance
        )
        
        if new_file or params_changed:
            # Atualizar os parâmetros no session_state
            st.session_state['frame_resize_factor'] = frame_resize_factor
            st.session_state['batch_size'] = batch_size
            st.session_state['frame_skip'] = frame_skip
            st.session_state['tolerance'] = tolerance
            
            # Diretório para salvar os vídeos processados
            output_dir = './processed_videos'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Gerar um nome de arquivo único para o vídeo processado
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"output_{timestamp}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            # Salvar o vídeo original temporariamente
            with tempfile.TemporaryDirectory() as temp_dir:
                video_path = os.path.join(temp_dir, uploaded_file.name)
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
                st.info("Carregando e processando o vídeo. Isso pode levar algum tempo dependendo do tamanho do vídeo.")
            
                # Carrega as faces conhecidas
                known_faces_dir = './known_faces'
                known_encodings, known_names = load_known_faces(known_faces_dir)
            
                if not len(known_encodings):
                    st.error("Nenhuma face conhecida foi carregada. Certifique-se de que o diretório 'known_faces' contém imagens válidas.")
                    return
            
                # Processa o vídeo
                with st.spinner('Processando o vídeo...'):
                    success = process_video(
                        video_path,
                        known_encodings,
                        known_names,
                        output_path=output_path,
                        tolerance=tolerance,
                        resize_factor=frame_resize_factor,
                        batch_size=batch_size,
                        frame_skip=frame_skip
                    )
            
                if success:
                    st.success("Processamento concluído!")
                    
                    # Ler o vídeo processado e armazenar no session_state
                    with open(output_path, 'rb') as f:
                        video_bytes = f.read()
                        st.session_state['processed_video_bytes'] = video_bytes
                        st.session_state['output_filename'] = output_filename
                        st.session_state['processed_video_name'] = uploaded_file.name
                else:
                    st.error("O processamento do vídeo falhou.")
        
        else:
            # Não é um novo arquivo e os parâmetros não mudaram, usar o vídeo processado anterior
            if st.session_state['processed_video_bytes'] is not None:
                st.success("Processamento concluído!")
            else:
                st.info("O vídeo foi carregado, mas ainda não foi processado.")
    
        # Exibe o vídeo processado e a opção de download se disponível
        if st.session_state['processed_video_bytes'] is not None:
            # Exibe o vídeo processado no Streamlit
            st.video(st.session_state['processed_video_bytes'])
            
            # Opção para baixar o vídeo processado
            st.download_button(
                label="Baixar Vídeo Processado",
                data=st.session_state['processed_video_bytes'],
                file_name=st.session_state['output_filename'],
                mime='video/mp4'
            )

if __name__ == "__main__":
    main()
