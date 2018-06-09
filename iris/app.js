const model = tf.sequential();
const EPOCH = 50;
const ITER = 10;
const LEARNING_RATE = 0.2;

// Layer 1
model.add(tf.layers.dense({
    units: 50,
    inputShape: [4],
    activation: 'sigmoid'
}));

// Layer 2
model.add(tf.layers.dense({
    units: 50,
    activation: 'sigmoid'
}))

// Output Layer
model.add(tf.layers.dense({
    units: 3,
    activation: 'softmax'
}))

const optimizer = tf.train.sgd(LEARNING_RATE);
model.compile({
    optimizer: optimizer,
    loss: 'meanSquaredError'
});

async function train() {
    for (let i = 0; i < ITER; i++) {
        let response = await model.fit(trainX, trainY, {
            shuffle: true,
            epochs: EPOCH
        });
        console.log(response.history.loss[0])
    }
}

async function main() {

    await train();

    let input = tf.tensor([
        [5.1, 3.5, 1.4, 0.2],
        [6.3, 3.3, 4.7, 1.6],
        [5.9, 3.0, 5.1, 1.8],
    ])
    let output = model.predict(input);

    output.print();
}

main();