const express = require('express');
const app = express();
const port = 3000;

app.use(express.static('web', { index: 'index-dev.html' }));
app.use('/web', express.static(__dirname + '/web'));
app.use('/dist', express.static(__dirname + '/dist'));
app.use('/sample_data', express.static(__dirname + '/sample_data'));
app.use('/db', express.static(__dirname + '/db'));

const message = `Example app listening on port ${port}. \n  click here: http://localhost:${port} \n  click here: http://localhost:${port}/web/covid19/index.html`

app.listen(port, () => console.log(message));
// app.listen(port, () => console.log(`Example app listening on port ${port} (click here: http://localhost:${port}).`));
// app.listen(port, () => console.log(`Example app listening on port ${port} (click here: http://localhost:${port}/web/covid19/index.html).`));
