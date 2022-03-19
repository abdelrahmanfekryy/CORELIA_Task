const TextArea = document.querySelector("#canvas textarea");
const TextResponse =  document.querySelector("#result p");
const SubmitButton = document.getElementById('submitbutton')
const ClearButton = document.getElementById('clearbutton')
const rects = document.querySelectorAll(".var");

let mymodel;
let tokenizer;
const Emotions = ['"Happy"','"Neutral"','"Sad"','"Angery"']

class L2 {

    static className = 'L2';

    constructor(config) {
       return tf.regularizers.l1l2(config)
    }
}
tf.serialization.registerClass(L2);



window.addEventListener("load" , (e) => {

    async function loadModel() {
    const model = await tf.loadLayersModel("https://raw.githubusercontent.com/abdelrahmanfekryy/CORELIA_Task/main/Emotion_Detection_From_Text/trained_models/model.json");
    return model
    }
    mymodel = loadModel();

    fetch('https://raw.githubusercontent.com/abdelrahmanfekryy/CORELIA_Task/main/Emotion_Detection_From_Text/trained_models/tokenizer.json')
    .then(response => response.json())
    .then(data => tokenizer = data)
    

    function ClearTextArea(){ 
        TextArea.value = ""
        TextResponse.innerHTML = 'Please Type A Text'
        rects.forEach(ClearChart)
    }

    var xx;
    function UpdateChart(item,index) {
        item.querySelector(".bar").setAttribute("style" ,`width:${xx[index]*100}%`);
        item.querySelector(".label p").innerHTML = (xx[index]*100).toFixed(2)
      }

      function ClearChart(item,index) {
        item.querySelector(".bar").setAttribute("style" ,`width:${0}%`);
        item.querySelector(".label p").innerHTML = 0.0
      }

      function Tokenize(item) {
        if (tokenizer[item] === undefined) {
            return 0;
        }
        else
        {
            return tokenizer[item];
        }
      }

    function ProcessData(e){
        let data = TextArea.value.toLowerCase().replace(/[^\w\s]/gi,'').replace(/\s+/g,' ').trim()
        console.log(data)
        data = data.split(" ").map(item => Tokenize(item))
        let tensor = tf.tensor1d(data).pad([[300 - data.length, 0]]).expandDims(0);
        mymodel.then(function (res) {
            console.log(res.predict(tensor).argMax(-1).dataSync()[0])
            let y_pred = res.predict(tensor)
            TextResponse.innerHTML = `The Predicted Emotion is ${Emotions[y_pred.argMax(-1).dataSync()[0]]}`
            xx = y_pred.dataSync()
            rects.forEach(UpdateChart)
        })
    }


  

    SubmitButton.addEventListener('click',ProcessData);
    ClearButton.addEventListener('click',ClearTextArea);
    
});



