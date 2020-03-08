#ver 1p0
import cv2,os
import math
import argparse
from agenet_helpers import load_to_IE, preprocessing
from agenet_inference import perform_inference


ALONE_KID_TIME_THREESHOLD =2 #in sec

def getfaces(frame,face_net , facenet_input_shape, conf_threshold =0.5):
    image=frame.copy()
    imageHeight=image.shape[0]
    imageWidth=image.shape[1]
    
    detected_face =perform_inference(face_net, 's', image, facenet_input_shape)
    faceBoxes=[]
    
    for i in range(detected_face['detection_out'].shape[2]):
        confidence =  detected_face['detection_out'][0,0,i,2]
        if confidence>=conf_threshold:
            x1=int(detected_face['detection_out'][0,0,i,3]*imageWidth)
            y1=int(detected_face['detection_out'][0,0,i,4]*imageHeight)
            x2=int(detected_face['detection_out'][0,0,i,5]*imageWidth)
            y2=int(detected_face['detection_out'][0,0,i,6]*imageHeight)
            faceBoxes.append([x1,y1,x2,y2])
            image=cv2.rectangle(image, (x1,y1), (x2,y2), (0,0,255), int(round(imageHeight/150)), 8)
    return image , faceBoxes
    

def detect_leftout_kid(occupants_age_category,fps):
    if not hasattr(detect_leftout_kid, "alone_kid_frame_counter"):
        detect_leftout_kid.alone_kid_frame_counter = 0
    alone_kid_frames_threeshold = ALONE_KID_TIME_THREESHOLD *fps
    if('kid' in occupants_age_category and (not 'adult' in occupants_age_category)):
        if(detect_leftout_kid.alone_kid_frame_counter>alone_kid_frames_threeshold):
            return 1
        else:
            detect_leftout_kid.alone_kid_frame_counter+=1
            return -1
    else:
        detect_leftout_kid.alone_kid_frame_counter=0
        return 0
parser=argparse.ArgumentParser()
parser.add_argument('--input')
curr_dir =os.getcwd()
cpu_ext_file_path = os.path.join(curr_dir,'.\cpu_extension_avx2.dll')
parser.add_argument('--cpu_ext',default=cpu_ext_file_path)
args=parser.parse_args()
CPU_EXTENSION = args.cpu_ext
#print('CPU EXtension',CPU_EXTENSION)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
#openvino Face detection model loading
face_net , facenet_input_shape=load_to_IE('face-detection-retail-0004.xml', CPU_EXTENSION)
#openvino agenet model loading
exec_net, input_shape = load_to_IE('age_net.xml', CPU_EXTENSION)


IS_IMAGE =False
if('.jpg' in args.input or '.bmp' in args.input or '.png' in args.input):
    IS_IMAGE = True 
    fps=0
    frame=cv2.imread(args.input)
    
else:
    video=cv2.VideoCapture(args.input if args.input else 0)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_height=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width=int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    file_path=args.input.split('.')
    del(file_path[len(file_path)-1])
    out_video_name = '.'.join(file_path)+'_inferred.avi'
    
    #print("FPS of video is ",fps)
    
    out_video=cv2.VideoWriter(out_video_name,cv2.CAP_OPENCV_MJPEG,cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
padding=20
cv2.namedWindow('Left Out Kid Detection in Car: Inferred Output',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Left Out Kid Detection in Car: Inferred Output', 600,450)
frame_no =1
while cv2.waitKey(1)<0:
    if(not IS_IMAGE):
     hasFrame,frame=video.read()
     if not hasFrame:
         cv2.waitKey(1)
         break
    
    resultImg,faceBoxes=getfaces(frame,face_net , facenet_input_shape)
    # print(type(frame))
    # cv2.imshow('Showing Frame',frame)
    # cv2.waitKey(0)
    if not faceBoxes:
        print("No face detected")
        
    occupants_age_category=[]
    i=1
    
    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        
        face_box_window ='Face Box '+str(i)
        i+=1
        
        #cv2.imshow(face_box_window,face)
        #openvino agenet model inference
        agePreds=perform_inference(exec_net, 's', face, input_shape)
        #print(agePreds)
        #print(agePreds['prob'])
        
        age=ageList[agePreds['prob'].argmax()]
        #print(f'Age: {age[1:-1]} years')
        if(ageList[0] in age or ageList[1] in age or ageList[2] in age): # Checking whether occupant is kid
            occupants_age_category.append('kid')
            current_occupant_category ='Kid'
        else:
            occupants_age_category.append('adult')
            current_occupant_category ='Adult'
        cv2.putText(resultImg,current_occupant_category , (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA) #f'{age}'
    kid_left_out=detect_leftout_kid(occupants_age_category,fps)
    if(kid_left_out==1):
        print('Frame no. {} Inference Output:  Warning :Detected alone kid'.format(frame_no))
        frame_height=resultImg.shape[0]
        cv2.putText(resultImg, 'Warning: Alone Kid Detected', (20, frame_height-20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2, cv2.LINE_AA)
        frame_no+=1
    elif(kid_left_out==0):
        print('Frame no. {} Inference Output:  Adult is present in car'.format(frame_no))
        frame_height=resultImg.shape[0]
        cv2.putText(resultImg, 'Adult is present in car', (20, frame_height-20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2, cv2.LINE_AA)
        frame_no+=1
    else:
        None 
    cv2.imshow('Left Out Kid Detection in Car: Inferred Output', resultImg)
    if(IS_IMAGE):
      cv2.waitKey(0)
      break
    else:  
        out_video.write(resultImg)
    

if(not IS_IMAGE):
    video.release()
    out_video.release()
cv2.destroyAllWindows() 
