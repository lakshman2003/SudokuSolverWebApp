const Imgs = document.querySelectorAll(".IMG");
const Input = document.getElementById("exInput");
const timer = document.getElementById("Timer");
const loading = document.getElementById("loading");
const Page = document.getElementById("Page");
// const solving = document.getElementById("solving");
const body = document.body;
var count = localStorage.getItem("Time");

if(count === null) count = 0;
else{
    timer.innerText = "Solved in: "+ count + "s";
}

function removeGreenBorder(){
    Imgs.forEach(img => {
        img.style.border = "3px white solid"
    })
}

function TimerStart(){
    count+=1;
    timer.innerText = "Solved in: "+ count + "s";
    
    // solving.innerText = "   "+ count + "s";
    // console.log(count)
    localStorage.setItem("Time",count)
}

function Start(){
    count = 0;
    loading.style.display = "flex";
    setInterval(TimerStart, 1000);
}

Imgs.forEach(img => img.addEventListener('click', () => {
        Input.value = img.src;
        removeGreenBorder();
        img.style.border = "3px #13af13 solid"
        console.log(img)
}))
