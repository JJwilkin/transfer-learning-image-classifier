import React, {useEffect}from 'react'
import './App.css';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs';
const Tensorset = require('tensorset');

function App() {


  async function save (originalClassifier, fileName) {
    let dataset = await  Tensorset.stringify(originalClassifier.getClassifierDataset());
    exportToJson(dataset);
  }

  async function exportToJson (objectData) {
    let filename = "model.json";
    let contentType = "application/json;charset=utf-8;";
    if (window.navigator && window.navigator.msSaveOrOpenBlob) {
      var blob = new Blob([decodeURIComponent(encodeURI(JSON.stringify(objectData)))], { type: contentType });
      navigator.msSaveOrOpenBlob(blob, filename);
    } else {
      var a = document.createElement('a');
      a.download = filename;
      a.href = 'data:' + contentType + ',' + encodeURIComponent(JSON.stringify(objectData));
      a.target = '_blank';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  }

  useEffect(() => {
    async function app() {
      
      let startRecording = false;
      let recordClass = "";
      let classifier = knnClassifier.create();
      console.log('Loading mobilenet..');
    
      // Load the model.
      const net = await mobilenet.load();
      console.log('Successfully loaded model');

      let existingJson = await (await fetch('./model.json')).json()
      
      const addClass = async (label) => {
        let className = label || document.getElementById('input').value;
        let btn = document.createElement("BUTTON");
        btn.innerHTML = `Add ${className}`;
        btn.onclick = function() { addExample(className); };
        document.getElementById("buttons").appendChild(btn); 
      }

      if (existingJson) {
        let dataset = await Tensorset.parse(existingJson);
        classifier.setClassifierDataset(dataset);
        let classes = Object.keys(classifier.getClassExampleCount());
        await classes.forEach((label) => {
          addClass(label);
        })
      }
    
      // Create an object from Tensorflow.js data API which could capture image 
      // from the web camera as Tensor.
      const webcam = await tf.data.webcam(document.getElementById('webcam'));
    
      // Reads an image from the webcam and associates it with a specific class
      // index.
      const addExample = async classId => {
        // Capture an image from the web camera.
        const img = await webcam.capture();
    
        // Get the intermediate activation of MobileNet 'conv_preds' and pass that
        // to the KNN classifier.
        const activation = net.infer(img, true);
    
        // Pass the intermediate activation to the classifier.
        classifier.addExample(activation, classId);
    
        // Dispose the tensor to release the memory.
        img.dispose();
      };

      document.getElementById('save').addEventListener('click', () => save(classifier, 'myModel'));
      document.getElementById('add').addEventListener('click', () => addClass());
      document.getElementById('count').addEventListener('click', () => console.log(classifier.getClassExampleCount()));
      document.getElementById("record").addEventListener("click", () => {
        console.log('recording...')
        recordClass = document.getElementById('recordInput').value;
        startRecording = true;
        setTimeout(() => {
          startRecording = false;
          console.log('done')
        }, 5000);
      });

      while (true) {
        if (classifier.getNumClasses() > 0) {
          const img = await webcam.capture();
          
          // Get the activation from mobilenet from the webcam.
          const activation = net.infer(img, 'conv_preds');
          // Get the most likely class and confidence from the classifier module.
          const result = await classifier.predictClass(activation);
    
          if (startRecording) {
            addExample(recordClass)
          }
          
          document.getElementById('console').innerText = `
            prediction: ${result.label}\n
            probability: ${result.confidences[result.label]}
          `;
    
          // Dispose the tensor to release the memory.
          img.dispose();
        }
    
        await tf.nextFrame();
      }
    }
    
    app();
  },[])
  return (
    <div className="App">
      <div style={{marginBottom:10}}>
        <video autoPlay playsInline muted id="webcam" width="224" height="224"></video>
        <div id="console"></div>
      </div>
      <button id="save">Save</button>
      <button id="count">Get Count</button>
      <input id='input'></input>
      <button id="add">Add Class</button>
      <input id='recordInput' placeholder="Class Name to Record"></input>
      <button id="record">Start Recording</button>
      <div id='buttons' style={{marginTop:10}}></div>
    </div>
  );
}

export default App;
