var textract = require('textract');
textract.fromFileWithPath('arabic_stopwords', function( error, text ) {})


/*
const util = require('util');

var methods = {};

var urls = [
    {year: '2016', link: 'http://www.url2016.pdf'},
    {year: '2015', link: 'http://www.url2015.pdf'}
];
var result = [];

const textractFromUrl = util.promisify(textract.fromUrl);


methods.download = function(req, res) {
    return extractText();
}

async function extractText() {
    try {
        var config = {
            preserveLineBreaks: true
        };
        for(let url of urls) {
            let text = await textractFromUrl(url.link, config);
            switch(url.year) {
                case '2015':
                    await extractTextType1(url, text);
                    break;

                case '2016':
                    await extractTextType2(url, text);
                    break;

                default:
                    console.log('Error: no switch case');
            }
        }
    }
    catch(err) {
        console.log('catch block');
        console.log(err);
    }
}
*/