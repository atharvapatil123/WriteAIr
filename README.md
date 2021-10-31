# WriteAIr

## Inspiration
During this pandemic, several professors have had to use teach virtually without access to a blackboard. Digital pens can be costly and writing on paper is difficult to record and stream. Thus, we have come up with a way for professors (and anyone else who wants to write and share their work) to write on AIR! Not only will this help professors who have gone virtual over the pandemic, but it also allows people to share their written work easily, no matter where they are!

## What it does
WriteAIr is a website which allows users to stream their writing. It uses HSV masking to detect a pen which the user writes with. Plus, users can select a wide range of options through hand gestures! The notes created can then be saved as images and uploaded on the server.

## How we built it
1. Django was used to create the website.
2. HSV masking was used to detect the marker of some specific colour.
3. A machine learning model was trained by us to detect hand gestures.
4. The machine learning model was trained on landmarks data of MediaPipe Hands calculated for images in our dataset which was then saved as a CSV file.

## Challenges we ran into
1. Converting images to PDF.
2. Figuring out how to create and train the model was quite the task.
3. Lack of existing dataset according to our needs. We had to create our own dataset of images.
4. Setting the sensitivity of the HSV masking so that background objects aren't detected.

## Accomplishments that we're proud of
1. We created our own dataset completely on our own!
2. We actually managed to train a model which had surprisingly good accuracy.
3. Integrated OpenCV with the gesture recognition model.
4. Hosted the project on a website using Django.

## What we learned
1. Most of us were completely new to machine learning, so we learnt a lot about how to create datasets, convert them into a proper .csv file, train the model and save it in a .h5 file.
2. Building a full-fledged website using Django.

## What's next for WriteAIr
1. Adding the project as a Google Meet extension or an alternative to Google Meet using sockets.
2. Increasing the accuracy of the pen detection code.
3. The code currently works for only right-handed users. An option to provide the service to left-handed users as well.
4. Integrating the website with the OpenCV part.
