const canvas = document.querySelector("#canvas canvas");
const ctx = canvas.getContext("2d");
const BrushCap = document.querySelector('.brushcap');
const TextResponse =  document.querySelector("#result p");
const SubmitButton = document.getElementById('submitbutton')
const ClearButton = document.getElementById('clearbutton')
const rects = document.querySelectorAll(".var");

const arabic_alpha = ['\u0623','\u0628','\u062A','\u062B','\u062C','\u062D','\u062E','\u062F','\u0630','\u0631','\u0632','\u0633','\u0634','\u0635','\u0636','\u0637','\u0638','\u0639','\u063A','\u0641','\u0642','\u0643','\u0644','\u0645','\u0646','\uFEEB','\u0648','\u064A'];
let painting = false;
let mymodel;

canvas.height = 400;
canvas.width = 400;

ctx.lineWidth = 30;
ctx.strokeStyle = 'red';
ctx.lineCap = 'round';

class L2 {

    static className = 'L2';

    constructor(config) {
       return tf.regularizers.l1l2(config)
    }
}
tf.serialization.registerClass(L2);



window.addEventListener("load" , (e) => {

    async function loadMobilenet() {
    const model = await tf.loadLayersModel("../trained_models/model.json");
    return model
    }
    mymodel = loadMobilenet();

    function StartPosition(e) {
        ctx.beginPath();
        painting = true;
        Draw(e);
    }

    function EndPosition() {
        painting = false;
    }

    function Draw(e) {
        if(!painting) return;
        ctx.lineTo(e.clientX - canvas.offsetLeft,e.clientY - canvas.offsetTop + document.documentElement.scrollTop)
        ctx.stroke();

        //for smoothness
        ctx.beginPath();
        ctx.lineTo(e.clientX - canvas.offsetLeft,e.clientY - canvas.offsetTop + document.documentElement.scrollTop);
        ctx.closePath();
    }


    function ClearCanvas(){ 
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        TextResponse.innerHTML = 'Please Sketch A Character'
        rects.forEach(ClearChart)
    }

    var xx;
    function UpdateChart(item,index) {
        item.querySelector(".bar").setAttribute("style" ,`width:${xx[index]*100}%`);
        item.querySelector(".label").innerHTML = (xx[index]*100).toFixed(2)
      }

      function ClearChart(item,index) {
        item.querySelector(".bar").setAttribute("style" ,`width:${0}%`);
        item.querySelector(".label").innerHTML = 0.0
      }


    function ProcessData(e){
        var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
        let values = tf.tensor3d(imageData.data, [canvas.height, canvas.width, 4]).resizeBilinear([32,32]).slice([0,0,0], [32,32,1]).div(tf.scalar(255)).expandDims(0);
        mymodel.then(function (res) {
            TextResponse.innerHTML = `The Predicted Character is ${arabic_alpha[res.predict(values).argMax(-1).dataSync()[0]]}`
            xx = res.predict(values).dataSync()
            rects.forEach(UpdateChart)
        })
    }
  

    function onMouseMove(e) {
        BrushCap.setAttribute("style" , `top: ${e.clientY + document.documentElement.scrollTop - 30/2}px;  left: ${e.clientX - 30/2}px; background-color:#ff0000; width: 30px; height: 30px`);
    }


    canvas.addEventListener('mousedown',StartPosition);
    document.addEventListener('mouseup', EndPosition);
    canvas.addEventListener('mousemove', Draw);
    canvas.addEventListener('mousemove', onMouseMove);
    SubmitButton.addEventListener('click',ProcessData);
    ClearButton.addEventListener('click',ClearCanvas);
    
});



