# AICTE_P2
P2: Attendance Management System Using Face Recognition 
Overview: This Python-based project automates student attendance tracking by leveraging face recognition technology. It employs OpenCV for face detection and recognition, along with a user-friendly GUI created with Tkinter. Key functionalities include real-time face recognition, image training, and efficient attendance record management.

Key Features Face Recognition: Automatically identifies students and marks attendance. Image Capture: Capture and save face images for training the recognition model. Manual Attendance: Option to manually update attendance records. CSV Export: Generate attendance reports in CSV format. Database Integration: Store attendance data in a MySQL database for easy retrieval. Technologies Used Programming: Python Libraries: OpenCV, Tkinter, NumPy, Pandas, Pillow Database: MySQL Installation Install Required Packages: Ensure Python is installed, then use:

bash Copy code pip install -r requirements.txt Set Up MySQL Database:

Create a MySQL database for attendance records. Update connection details in the code as needed. Download Haarcascade:

Obtain the haarcascade_frontalface_default.xml file from OpenCV's GitHub repository. Place it in the project directory. Usage Capture Images:

Run main_Run.py to open the GUI. Enter the student's enrollment number and name. Click "Take Images" to capture face images. Train the Model:

After capturing images, click "Train Images" to build the face recognition model. Automatic Attendance:

Select "Automatic Attendance" to initiate real-time face recognition using a webcam. Manual Attendance:

Use the "Manually Fill Attendance" option to update records manually. View Registered Students:

Access the admin panel to review registered student details. Directory Structure graphql Copy code Attendance Management System Using Face Recognition/ ├── TrainingImage/ # Stores captured training images ├── TrainingImageLabel/ # Saves the trained model ├── StudentDetails/ # CSV files with student details ├── Attendance/ # Stores attendance records ├── haarcascade_frontalface_default.xml # Haarcascade file for detection ├── requirements.txt # Python dependencies ├── main_Run.py # Main application file ├── training.py # Model training script ├── testing.py # Face recognition testing script ├── mini_app.py # GUI for image capture ├── app.py # Streamlit app for attendance visualization └── README.md # Project documentation Contributing Contributions are encouraged! Suggestions for new features or improvements can be submitted as pull requests. License Licensed under the MIT License. See the LICENSE file for details. Acknowledgments Special thanks to:

OpenCV for face detection and recognition. Tkinter for GUI development. NumPy and Pandas for data handling. MySQL for database management. For inquiries or support, contact [your email].
