import face_recognition as fc
import cv2 as cv
import numpy as np
import os
import pickle
from datetime import datetime

def new_person():
  """Fuction adds a new person into its database."""
  name = 'interviewer'
  filename = 'interviewer.jpeg'
  image = cv.imread(filename)
  encodings =[]
  conv_image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
  encodings.append(fc.face_encodings(conv_image , fc.face_locations(image))[0])
  with open('knownpeeps/'+name+'.pickle', 'wb') as handle:
    pickle.dump(encodings, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_all_encodings():
  """Results in finding encodings."""
  encodings = {}
  names = []
  for i in os.listdir('knownpeeps/'):
    name = i.replace('.pickle','')
    with open('knownpeeps/'+i, 'rb') as handle:
      encodings[name] = pickle.load(handle)
    names.append(name)
  return encodings,names

def get_video():
  encodings,names = get_all_encodings()
  vid = cv.VideoCapture('video.mp4')
  lis =[]
  while vid.isOpened():
    return_m , image1 = vid.read()
    if(return_m):
      cv.imshow('Frame',image1)
      image = cv.cvtColor(image1,cv.COLOR_BGR2RGB)
      face_locations = fc.face_locations(image)#,model='cnn')
      face_encodings = fc.face_encodings(image,face_locations)
      for i,face_location in zip(face_encodings,face_locations):
        for x in names:
          results = fc.compare_faces(np.array(encodings[x]),i,0.7)
          if True in results:
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            cv.rectangle(image, top_left, bottom_right, 3)
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)
            cv.rectangle(image,top_left,bottom_right,cv.FILLED)
            cv.putText(image,x,(face_location[3]+10,face_location[2]+5),cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            image = cv.cvtColor(image,cv.COLOR_RGB2BGR)
            cv.imshow('faces',image)
            lis.append([x,datetime.now().strftime("%H%M%S")])
          if cv.waitKey(0) and 0xFF == ord('q'):
            break
      if cv.waitKey(0) and 0xFF == ord('q'):
        break
        
    else :
      break
  vid.release()
  cv.destroyAllWindows()
  return lis

if __name__ == '__main__':
  print(get_video())
  #new_person()
  #check_photo()
