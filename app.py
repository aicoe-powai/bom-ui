# Python In-built packages
from pathlib import Path
import PIL
from PIL import Image
import PIL.Image
import cv2
# External packages
import streamlit as st
import tempfile
import numpy as np
import pandas as pd

# Local Modules
import settings
import helper
from streamlit_extras.app_logo import add_logo

#global param
global_check = False
frame_global_output = {}
bom_name_id = { 
"ACOS" : ["ACOS","SPE0101172133"],
"EMIS" : ["Emis Filter","IPMS1064C2A01"],
"ETHSW" : ["Ethernet Switch","IPMS105146201"],
"EXNWC" : ["Extended RJ45 Network Connector","SPE0101186257"],
"FAN" : ["Fan","SPE0101186501"],
"DTTB" : ["Mini Terminal Block","SPE0101173665"],
"THMT" : ["Thermostat","IPMS1108KQW01"],
"DSBC" : ["DSUB Backshell Connector","IPMS105AWRU01"],
"MCB" : ["MCB","-"],
"FRM" : ["Fuse Relay Module","IPMS1019O3S01"],
"IORT" : ["Input Output Relay Terminal","IPMS1091WAZ01"],
"PLI" : ["Power Light Indicator","-"],
"ISOC" : ["Isolated Converter","DD900067-6"],
"RSR" : ["RS232 Repeater","-"]

} #com_class_name , com_name, bom_item_code

# Setting page layout
st.set_page_config(
    page_title="L&T PES BOM Verification",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
# def logo():
#     add_logo(settings.LOGO_PATH, height=300)


# Main page heading
st.title("L&T PES BOM Verification")

with st.sidebar:
    st.image("C:\\Users\\20372227\\Desktop\\Model Workspace\\ui\\assets\\logo_new.jpg", width=250)

confidence = 0.25
model_path = settings.DETECTION_MODEL
# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Setup")
source_radio = st.sidebar.radio(
    "**Select Source**", settings.SOURCES_LIST)

source_component = st.sidebar.radio(
    "**Select Component**", settings.SOURCES_COMPONENT)

source_sections = st.sidebar.radio(
    "**Select Sections**", settings.SOURCES_SECTIONS
)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                # default_image = PIL.Image.open(default_image_path)
                default_image = cv2.imread(default_image_path)
                st.image(default_image_path, caption="Input Image",
                          width = 480)
            else:
                uploaded_image = PIL.Image.open(source_img)
                uploaded_image = uploaded_image.crop((0,0,2448,2448))
                uploaded_image = uploaded_image.resize((480,480))

                st.image(source_img, caption="Uploaded Image",
                          width = 480 )
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     width=480)
            #use_column_width=True
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         width=480)
                #use_column_width=True
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")
    #Table logic
    with col2:
        # Static table
        data = {
            "S.No" : [1,2,3],
            "Component Name": ["Fan","Acos","Ethernet Switch"],
            "Component Class" : ["FAN" , "ACOS" , "ETHSW"],
            "BOM Class" : ["SPE0101186501","SPE0101172133","IPMS105146201"],
            "Quantity" : [1,2,3]
        }
        df = pd.DataFrame(data)
        st.table(df)

elif source_radio == settings.VIDEO:
    source_video = st.sidebar.file_uploader(
        "Choose a video...", type=("mp4", "avi", "mov", "wmv"))
    
    if 'image_index' not in st.session_state:
        st.session_state.image_index = 0
            
    if 'frames' not in st.session_state:
        st.session_state.frames = []

    def extract_frames_from_video(video_path):
        frames = []
        vid_cap = cv2.VideoCapture(video_path)
        
        while True:
            success, frame = vid_cap.read()
            if not success:
                break
            # Convert frame to RGB and store as PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
        
        vid_cap.release()
        return frames
    
    with st.container():
        col1, col2 = st.columns(2)
        # print(col1, col2)

        with col1:
            try:
                if source_video is None:
                    default_video_path = str(settings.DEFAULT_VIDEO)
                    st.video(default_video_path, format="video/mp4")
                    # st.write("Input Video")
                    # Add a caption below the video with custom styling
                    caption = """
                    <div style="text-align: center; color: black; font-size: 25px;">
                        Input Video
                    </div>
                    """

                    st.markdown(caption, unsafe_allow_html=True)
                else:
                    
                    # temp_file = tempfile.NamedTemporaryFile(delete=False)
                    # temp_file.write(source_video.read())
                    # vid_cap = cv2.VideoCapture(temp_file.name)

                    # # Define a temporary output file to store the resized video
                    # # out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    # out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')

                    # fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    # # Define the codec and create VideoWriter object for the resized video
                    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
                    # # out = cv2.VideoWriter(out_file.name, fourcc, fps, (480, 480))
                    # out = cv2.VideoWriter(out_file.name, fourcc, fps, (480,480))
                    # while True:
                    #     success, frame = vid_cap.read()
                        
                    #     # if success:
                    #     #     h, w, _ = frame.shape
                    #     #     min_dim = min(h, w)
                    #     #     if h > w:
                    #     #         # Crop the top
                    #     #         start_y = 0
                    #     #         cropped_frame = frame[start_y:min_dim, 0:w] 
                    #     #     else:
                    #     #         # Center crop
                    #     #         start_x = (w - min_dim) // 2
                    #     #         cropped_frame = frame[0:h, start_x:start_x + min_dim]
                    #     #     frame_resized = cv2.resize(cropped_frame, (480, 480))
                    #     out.write(frame)
                    #     if not success:
                    #         vid_cap.release()
                    #         out.release()
                    #         break

                    st.video(source_video, format = "video/mp4") #Uploaded video
                                    
                    # tfile = tempfile.NamedTemporaryFile(delete=False)
                    # tfile.write(uploaded_file.read())
                    # # Open video file with OpenCV
                    # cap = cv2.VideoCapture(tfile.name)
                    # # Get original video's width, height, and FPS
                    # original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    # original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                
            
                    # # Process video frame by frame
                    # while cap.isOpened():
                    #     ret, frame = cap.read()
                    #     if not ret:
                    #         break
                    #     # Resize the frame to 480x480
                    #     resized_frame = cv2.resize(frame, (480, 480))
                
                    #     # Write the resized frame to the output video
                    #     out.write(resized_frame)
                
                    # # Release resources
                    # cap.release()
                    # out.release()
                
                    # # Display the resized video in Streamlit
                    # st.text("Video resizing completed. Displaying resized video:")
                    # st.video(out_file.name)

                    

            except Exception as ex:
                st.error("Error occurred while opening the video.")
                st.error(ex)

            caption = """
                    
                    """
            # Giving some empty space between the Input Video and Detected video
            st.markdown(caption, unsafe_allow_html=True)

            #Having the detected video below the input video
            
            # else:
            #     # is_display_tracker, tracker = helper.display_tracker_options();
            #     if st.sidebar.button('Detect Objects'):
            #         temp_file = tempfile.NamedTemporaryFile(delete=False)
            #         temp_file.write(source_video.read());
            #         try:
            #             vid_cap = cv2.VideoCapture(
            #                 str(temp_file.name))
                        
            #             st_frame = st.empty()
            #             while True:
            #                 success, frame = vid_cap.read()
            #                 if success:
            #                     # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            #                     # frame = frame[0:2448,0:2448] #cropping
            #                     # frame = cv2.resize(frame, (480, 480)) #resizing                        
            #                     h, w, _ = frame.shape
            #                     min_dim = min(h, w)
            #                     if h > w:
            #                         # Crop the top
            #                         start_y = 0
            #                         cropped_frame = frame[start_y:min_dim, 0:w] 
            #                     else:
            #                         # Center crop
            #                         start_x = (w - min_dim) // 2
            #                         cropped_frame = frame[0:h, start_x:start_x + min_dim]
            #                     frame_resized = cv2.resize(cropped_frame, (480, 480))
            #                     # frame = cv2.resize(frame, (720, int(720*(9/16))))  # Resize if needed
            #                     helper._display_detected_frames(confidence,
            #                                             model,
            #                                             st_frame,
            #                                             frame_resized
            #                                             # is_display_tracker,
            #                                             # tracker
            #                                             )
            #                 else:
            #                     vid_cap.release()
            #                     break
            #         except Exception as e:
            #             print("err", e)
            #             st.sidebar.error("Error loading video: " + str(e))
            
        # helper.play_stored_video(confidence, model)

        

        # Table logic
        with col2:
            # Static table
            # data = {
            #     "S.No" : [1,2,3],
            #     "Component Name": ["Fan","Acos","Ethernet Switch"],
            #     "Component Class" : ["FAN" , "ACOS" , "ETHSW"],
            #     "BOM Class" : ["SPE0101186501","SPE0101172133","IPMS105146201"],
            #     "Quantity" : [1,2,3]
            # }
            # df = pd.DataFrame(data)
            # st.table(df)

            if source_video is None:
                default_detected_video_path = str(settings.DEFAULT_DETECT_VIDEO)
                st.video(default_detected_video_path, format="video/mp4")
                caption = """
                    <div style="text-align: center; color: black; font-size: 25px;">
                        Detected Video
                    </div>
                    """

                st.markdown(caption, unsafe_allow_html=True)
            else:
                if st.sidebar.button('Detect Objects'):
                    temp_file = tempfile.NamedTemporaryFile(delete=False)
                    temp_file.write(source_video.read())
                    try:
                        vid_cap = cv2.VideoCapture(str(temp_file.name))
                        st_frame = st.empty()
                        frame_re_ini = 0
                        while True:
                            success, frame = vid_cap.read()
                            if success:
                                h, w, _ = frame.shape
                                min_dim = min(h, w)
                                if h > w:
                                    # Crop the top
                                    start_y = 0
                                    cropped_frame = frame[start_y:min_dim, 0:w] 
                                else:
                                    # Center crop
                                    start_x = (w - min_dim) // 2
                                    cropped_frame = frame[0:h, start_x:start_x + min_dim]
                                frame_resized = cv2.resize(cropped_frame, (480, 480))
                                frame_global_output = helper._display_detected_frames(confidence,model,st_frame,frame_resized, frame_re_ini)
                                frame_re_ini+=1
                            else:
                                vid_cap.release()
                                global_check = True
                                break
                    except Exception as e:
                        print("Error:", e)
                        st.sidebar.error("Error loading video: " + str(e))

                #Dynamic table
            if global_check == True:
                # Initialize a dictionary to accumulate quantities
                component_totals = {}
                for frame_index, counts in frame_global_output.items():
                    for class_name, count in counts.items():
                        # Accumulate the quantity for each component class
                        if class_name in component_totals:
                            component_totals[class_name] += count
                        else:
                            component_totals[class_name] = count
                table_data = []
                s_no = 1  
                for class_name, total_quantity in component_totals.items():
                    # Check if class_name is in bom_name_id to get the Component Name and BOM Item Code
                    if class_name in bom_name_id:
                        component_name, bom_item_code = bom_name_id[class_name]
                    else:
                        component_name = None
                        bom_item_code = None

                    # Append a new row for each unique component class
                    table_data.append({
                        "S.No": s_no,
                        "Component Class": class_name,
                        "Component Name": component_name,
                        "BOM Item Code":bom_item_code,
                        "Quantity": total_quantity
                    })
                    s_no += 1
                frame_global_output.clear()
                df = pd.DataFrame(table_data)
                st.table(df)

            

            # def process_video_frames(video_path, confidence, model, is_display_tracker, tracker):
            #     frames = []
            #     vid_cap = cv2.VideoCapture(video_path)
            #     st_frame = st.empty()
            #     while True:
            #         success, frame = vid_cap.read()
            #         if success:
            #             # Process the frame and append it to the list
            #             processed_frame = helper._display_detected_frames(confidence, model, st_frame, frame, is_display_tracker, tracker)
            #             frames.append(processed_frame)
            #         else:
            #             vid_cap.release()
            #             break
                
            #     return frames

            ## Returning frame by frame
            # with col2:
            #     if source_video is None:
            #         default_detected_video_path = str(settings.DEFAULT_DETECT_VIDEO)
            #         st.video(default_detected_video_path, format="video/mp4")
            #     else:
            #         is_display_tracker, tracker = helper.display_tracker_options()
            #         if st.sidebar.button('Detect Objects'):
            #             temp_file = tempfile.NamedTemporaryFile(delete=False)
            #             temp_file.write(source_video.read())
            #             video_path = str(temp_file.name)
                    
            #             try:
            #                 # Get all frames with detection
            #                 # frames = process_video_frames(video_path, confidence, model, is_display_tracker, tracker)
            #                 # frames = process_video_frames(video_path, confidence, model, is_display_tracker, tracker)
                            
            #                 # Extract frames from the video
            #                 st.session_state.frames = extract_frames_from_video(video_path)

            #                 # Check if there are frames to display
            #                 if st.session_state.frames:
            #                     # Display the current frame with detections
            #                     st_frame = st.empty()
            #                     # # Display the current frame based on the session state
            #                     # st.image(frames[st.session_state.image_index], caption=f"Frame {st.session_state.image_index + 1}", use_column_width=True)
                                
            #                     current_frame = np.array(st.session_state.frames[st.session_state.image_index])
            #                     helper._display_detected_frames(confidence, model, st_frame, current_frame, is_display_tracking=None, tracker=None)

            #                     # Navigation buttons
            #                     col1, col2 = st.columns([1, 1])

            #                     with col1:
            #                         if st.button("Previous Frame") and st.session_state.image_index > 0:
            #                             st.session_state.image_index -= 1
            #                             current_frame = np.array(st.session_state.frames[st.session_state.image_index])
            #                             helper._display_detected_frames(confidence, model, st_frame, current_frame, is_display_tracking=None, tracker=None)
                                
            #                     with col2:  
            #                         if st.button("Next Frame") and st.session_state.image_index < len(st.session_state.frames) - 1:
            #                             st.session_state.image_index += 1
            #                             current_frame = np.array(st.session_state.frames[st.session_state.image_index])
            #                             helper._display_detected_frames(confidence, model, st_frame, current_frame, is_display_tracking=None, tracker=None)
            #                 else:
            #                     st.write("No frames to display.")
            #             except Exception as e:
            #                 print("Error", e)
            #                 st.sidebar.error("Error loading video: " + str(e))

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)
