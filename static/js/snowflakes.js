// Marching Squares
// Coding in the Cabana
// The Coding Train / Daniel Shiffman
// https://thecodingtrain.com/challenges/coding-in-the-cabana/005-marching-squares.html
// https://youtu.be/0ZONMNUKTfU
// p5 port: https://editor.p5js.org/codingtrain/sketches/g6-zufS8c

let field = [];
let rez = 15;
let cols, rows;
let increment = 0.1;
let zoff = 0;
let noise;
let canvas;

// function windowResized() {
//   //console.log('resized');
//   canvas.resizeCanvas(windowWidth, windowHeight);
// }

function setup() {
    canvas = createCanvas(windowWidth, 4.2*windowHeight); //make it an object
    canvas.position(0,0);
    canvas.style('z-index','-1');
  noise = new OpenSimplexNoise(Date.now());
  cols = 1 + width / rez;
  rows = 1 + height / rez;
  for (let i = 0; i < cols; i++) {
    let k = [];
    for (let j = 0; j < rows; j++) {
      k.push(0);
    }
    field.push(k);
  }
}

function drawLine(v1, v2) {
  stroke('#f92672');
  line(v1.x, v1.y, v2.x, v2.y);
}

function draw() {
  // background(0); 
  clear();
  let xoff = 0;
  for (let i = 0; i < cols; i++) {
    xoff += increment;
    let yoff = 0;
    for (let j = 0; j < rows; j++) {
      field[i][j] = float(noise.noise3D(xoff, yoff, zoff));
      yoff += increment;
    }
  }
  zoff += 0.03;




  //for (let i = 0; i < cols; i++) {
  //  for (let j = 0; j < rows; j++) {
  //    fill(field[i][j]*255);
  //    noStroke();
  //    rect(i*rez, j*rez, rez, rez);
  //  }
  //}

  for (let i = 0; i < cols-1; i++) {
    for (let j = 0; j < rows-1; j++) {
      let x = i * rez;
      let y = j * rez;
      dx = 0.5;//field[i][j] / (field[i][j]+field[i+1][j]);
      let a = createVector(x + rez * dx, y            );
      dx = 0.5;//field[i+1][j+1] / (field[i+1][j]+field[i+1][j+1]);
      let b = createVector(x + rez, y + rez * dx);
      dx = 0.5;//field[i][j+1] / (field[i][j+1]+field[i+1][j+1]);
      let c = createVector(x + rez * dx, y + rez      );
      dx = 0.5;//field[i][j+1] / (field[i][j]+field[i][j+1]);
      let d = createVector(x, y + rez * dx);
      let state = getState(ceil(field[i][j]), ceil(field[i+1][j]), 
        ceil(field[i+1][j+1]), ceil(field[i][j+1]));
      stroke(255);
      strokeWeight(1);
      switch (state) {
      case 1:  
        drawLine(c, d);
        break;
      case 2:  
        drawLine(b, c);
        break;
      case 3:  
        drawLine(b, d);
        break;
      case 4:  
        drawLine(a, b);
        break;
      case 5:  
        drawLine(a, d);
        drawLine(b, c);
        break;
      case 6:  
        drawLine(a, c);
        break;
      case 7:  
        drawLine(a, d);
        break;
      case 8:  
        drawLine(a, d);
        break;
      case 9:  
        drawLine(a, c);
        break;
      case 10: 
        drawLine(a, b);
        drawLine(c, d);
        break;
      case 11: 
        drawLine(a, b);
        break;
      case 12: 
        drawLine(b, d);
        break;
      case 13: 
        drawLine(b, c);
        break;
      case 14: 
        drawLine(c, d);
        break;
      }
    }
  }
}

function getState(a, b, c, d) {
  return a * 8 + b * 4  + c * 2 + d * 1;
}
