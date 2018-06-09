// Convert csv to array

const fs = require('fs');
fs.readFile('data.csv', 'utf8', (err, data) => {
    data = data.split('\n');
    data = data.splice(1, data.length - 1);
    
    let xs = [];
    let ys = [];
    data.forEach((row) => {
        row = row.split(',');
        let info = row.splice(1, 4);
        info = info.map(d => Number(d))
        xs.push(info);

        let label = row[1] == "Iris-setosa" ? [1, 0, 0] : row[1] == "Iris-versicolor" ? [0, 1, 0] : [0, 0, 1];
        ys.push(label)
    });

    // fs.writeFileSync('data.js', JSON.stringify(xs, null, 4), 'utf8');
    fs.writeFileSync('labels.js', JSON.stringify(ys, null, 4), 'utf8');
})