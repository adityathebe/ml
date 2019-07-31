let x_vals = []
let y_vals = []
let a, b, c;

let slider;
const learning_rate = 0.5;
const optimizer = tf.train.sgd(learning_rate);

function setup() {
    createCanvas(1200, 580);
    frameRate(60);
    textSize(25);

    slider = createSlider(0.1, 0.7, 0.5, 0.1)
    a = tf.variable(tf.scalar(random(1)));
    b = tf.variable(tf.scalar(random(1)));
    c = tf.variable(tf.scalar(random(1)));
}

function draw() {
    background(51);
    fill(255);

    if (x_vals.length == 0) return;
    tf.tidy(() => {

        // Plot Points
        noStroke();
        x_vals.forEach((x, index) => { ellipse(x, y_vals[index], 5, 5) })

        // Normalize Points and Train
        const normalizedX = x_vals.map(x => x / width);
        const normalizedY = y_vals.map(y => y / height);
        const xt = tf.tensor1d(normalizedX);
        const yt = tf.tensor1d(normalizedY);
        optimizer.minimize(() => loss(predict(xt), yt));
        
        // Adjust Learning Rate based on slider value
        optimizer.setLearningRate(slider.value());
        
        // Display stats
        let loss_value = loss(predict(xt), yt).dataSync()[0].toFixed(4);
        text("Loss: " + loss_value, width * 0.05, height * 0.10);
        text("Learning Rate: " + optimizer.learningRate, width * 0.05, height * 0.15);
        let equation = `${a.dataSync()[0].toFixed(2)}x^2 + ${b.dataSync()[0].toFixed(2)}y + ${c.dataSync()[0].toFixed(2)} = 0`;
        text(equation, width * 0.05, height * 0.2)

        // Draw the Curve
        strokeWeight(2);
        stroke('red');
        const xs = tf.range(0, 1, 0.05);
        const ys = predict(xs).dataSync();
        beginShape()
        noFill();
        for (let i = 0; i < ys.length; i++) {
            let x = map(xs.dataSync()[i], 0, 1 , 0, width);
            let y = map(ys[i], 0, 1, 0, height)
            vertex(x, y);
        }
        endShape();
    });
}

function predict(x) {
    return a.mul(x.square()).add(b.mul(x)).add(c);
}

function loss(pred, labels) {
    return pred.sub(labels).square().mean();
}

function mousePressed(data) {
    if (data.button != 0) return; // Only Left Click Allowed
    if (mouseX > width || mouseX < 0) return;
    if (mouseY > height || mouseY < 0) return;

    x_vals.push(mouseX);
    y_vals.push(mouseY);
}