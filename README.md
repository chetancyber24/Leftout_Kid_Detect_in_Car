# How to run Python Program in Windows 10

 - Clone this Github reprository.
 - Open command prompt and initialize Openvino environment (run
   setupvars.bat).
 - Navigate to Github reprository folder in command prompt.
 - Run this command on command prompt: **python Left_out_kid_detect_openvino.py  --input .\demo_video.mp4   --cpu_ext .\cpu_extension_avx2.dll**
 - ***--input path_to_input_video_file*** arg parameter is used to provide input video file(car cabin view) to perform inference on it.
 - ***--cpu_ext path_to_cpu_extension_file*** arg parameter is used to provide cpu extension file. It is processor and OS dependent file so provide cpu extension file accordingly to your system configuration. 
 - Expected output will be:

   On newly opened opencv window video will play along with          classification of car        occupants as either kid or adult. It    will also   have warning text embedded on video when  alone kid is    present after   threshold amount of time (set to 2 sec here,          ALONE_KID_TIME_THREESHOLD parameter inside Python script.) (see below section **Output snapshots**). If any key pressed  during video inferring    , program will end.

	Final inferred output video also will be saved in same input video directory named as demo_video_inferred.mp4
	
    Also in command prompt frame by frame detection status is printed.
    
	

 - Above program is tested on Windows 10 environment. For Linux openvino
   setup  and cpu extension, please follow Openvino documentation.



# Output Snapshots

![Command prompt output](https://github.com/chetancyber24/Leftout_Kid_Detect_in_Car/blob/master/images/snapshot1.jpg)
 **Snapshot 1 : Python command prompt output showing each frame inferred status(Adult present or not).** 

![Inferred Frame with adult](https://github.com/chetancyber24/Leftout_Kid_Detect_in_Car/blob/master/images/snapshot2.jpg)
**Snapshot 2: Frame inferred in video where adult is present in car with kid.**

![Inferred Frame with alone kid](https://github.com/chetancyber24/Leftout_Kid_Detect_in_Car/blob/master/images/snapshot3.jpg)
 **Snapshot 3: Frame inferred in video where alone kid is present in car with warning embedded.**


# Demo Video
Python Program Demo Video can be accessed here : https://www.youtube.com/watch?v=978tXgmopO4&feature=youtu.be

 


# Problem Statement

Child heatstroke car death is big safety issue and it caused significant no of children death worldwide.  In 2019, there are **53 children deaths** due to hot car death alone in US (reference [https://www.kidsandcars.org/how-kids-get-hurt/heat-stroke/](https://www.kidsandcars.org/how-kids-get-hurt/heat-stroke/)) .  On average there are 38 deaths per year and **940 children death** since 1990 in US alone, worldwide number will be more.



# Solution(Idea)

Our  idea to solve above problem is to use in car camera to detect alone left out kid(s) with help of computer vision and deep learning model. Below is pictorial representation (**Flow Chart Section**) of our idea to detect left out kid in car. We are using camera to detect occupants age and deciding whether there are **only kid(s)** present in car for more than threshold time, if yes, we raise alarm i.e send sms or app notifications to parents  either using Car V2X technology(Connected car) or in-built modem with camera. It can also raise alarm to emergency response team (i.e. 911).

To detect occupants age, first we use face detection deep learning model to detect face and get its associated bounding box. Then we cropped face of each occupant and feed it to age prediction deep learning model to predict age of each occupant in car. Once we have predicted age of all occupants, we categorized each occupant to kid (less than or equal to 12 years) or adult (greater than 12 years) in given frame. If we detect only kid(s) with no adult present in car, we raise alarm after programmable threshold amount of time elapsed (say 15min). In Python program we didn’t implemented sms service (paid service) or app notification, instead to show alarm we just embedded frame with warning text. Messaging service can be implemented with **Twilio API**

For face detect model, Openvino pretrained model **face-detection-retail-0004** is used and for age prediction we used **[Gil Levi and Tal Hassner Age Classification](https://talhassner.github.io/home/publication/2015_CVPR)** Using Convolutional Neural Network. Age prediction model is available as Caffe model, we converted it to Openvino IR format using model optimizer.

# Flow Chart
![Algorithm Flow Chart](https://github.com/chetancyber24/Leftout_Kid_Detect_in_Car/blob/master/flow_chart.png)







# Known Issue, Possible Solutions & Further enhancement

 - In some random frame of video age prediction model make wrong
   prediction and classify adult into kid (or vice versa).  To supress
   false alarm, kid alone alarm will be raised after certain amount of
   time where consecutive frame is detected kid with no adult present.
   In python program, this threshold parameter is determined by
   ALONE_KID_TIME_THREESHOLD it is set to 2sec in program because of
   short demo video. In real life this can be set to slightly longer
   time (i.e. 15 mins).
  
 - Currently when kid is detected without adult, as alarm video is   
   embedded with warning text. In real life this should trigger warning 
   sms or app notification to parents’ phone or 911 emergency responder 
   using V2X technology (connected vehicle).
 - This further can be enhanced to disable specific airbag when small
   kid is detected.
 - Camera mounted on rear view mirror capturing view of inside car cabin
   will not be able to capture child sat in **rear facing car seat**. To
   solve this issue one more camera(red camera in below snapshot), need
   to mount on near back windshield to capture kid sitting in rear
   facing seat.
   ![Camera in back windshield](https://github.com/chetancyber24/Leftout_Kid_Detect_in_Car/blob/master/images/snapshot4.jpg)
**Figure: Camera mounted on near back windshield to capture kid seating in rear facing car seat.**


# Google Colab Project(Alternative)
As a different approach we also tried installing OpenVino in Google Colab and using Res10 Single Shot Detector Caffe Model for face detection and the same Age Classification Network.

Steps to run the project in Google Colab:
 - Access the Folder for the project from the following link:
Child_Lost_In_Car
 - Open the colab notebook: ‘ Detecting_Kids_Left_Alone_In_a_Car.ipynb’
 - Change the RunType to GPU
Menubar → RunTime → Change runtime type
 - Execute each cell by cell ( cntrl + enter) or (alt+enter) or `'/>'` button near each cells
Or Menubar → RunTime → run all
   
 - Please Note: After running or the execution of the first cell for
   mounting the Google drive in  either step 3 or step 4 you choose, you will be prompted to click on the link generated to   authenticate the mounting of the Google Drive
 - Read through the Colab Notebook to download the output file and see the output.

# Project Members
Chetan Verma, Lakshmi Prasannakumar & Saikat Pandit

# References:

 - Udacity Intel® Edge AI Scholarship Foundation Course
 - Intel OpenVino Documentation
 - Age detection Model
   ([https://data-flair.training/blogs/python-project-gender-age-detection/](https://data-flair.training/blogs/python-project-gender-age-detection/))
 - Sample videos (shutterstock.com)




