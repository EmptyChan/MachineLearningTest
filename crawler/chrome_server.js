const express = require('express') 
const bodyParser = require("body-parser");
const puppeteer = require('puppeteer'); 
const request = require('request');
const FdfsClient = require('fdfs');

const fdfs_host = '172.17.128.189';
const proxy_url = 'http://172.16.128.130:5000/get';//'http://localhost:5555/random';//'http://172.16.128.130:5000/get';
//const proxy_url = 'http://localhost:5555/random';
const fdfs_port = 22122;

const app = express(); 



var fdfs = new FdfsClient({
    // tracker servers
    trackers: [
        {
            host: fdfs_host,
            port: fdfs_port
        }
    ],
    // 默认超时时间10s
    timeout: 10000,
    // 默认后缀
    // 当获取不到文件后缀时使用
    defaultExt: 'png',
    // charset默认utf8
    charset: 'utf8'
});


app.use(bodyParser.urlencoded({extended: false}));
app.use(bodyParser.json());

app.post('/', function (req, res) { 
	var url=req.body.url; 
	console.log(url);
	var type=req.body.type; 
	var jsonObjectStr=req.body.jsonObject; 
	var areaSelectStr=req.body.areaSelect;
	var cookiesStr=req.body.cookies;
	
	var requestAction = function (url) {
		return new Promise(function (resolve, reject) {
			request({url: url}, function (error, response, body) {
				if (error) return resolve('');
				resolve(body);
			})
		});
	};
		
	(async() => { 	
		var proxy= await requestAction(proxy_url);
		var browser;
		if (proxy!=''){
			console.log(0);
			browser = await puppeteer.launch({headless: false,// 关闭headless模式, 会打开浏览器
			ignoreHTTPSErrors: true,//如果是访问https页面 此属性会忽略https错误
			//设置超时时间
			timeout: 15000,
			args: ['--no-sandbox', '--disable-setuid-sandbox', '--proxy-server='+proxy]}); 
			console.log('browser:'+proxy);
		}
		else {
			console.log(1);
			browser = await puppeteer.launch({headless: false,
			ignoreHTTPSErrors: true,//如果是访问https页面 此属性会忽略https错误
			//设置超时时间
			timeout: 15000,
			args: ['--no-sandbox', '--disable-setuid-sandbox']}); 
		}
		const page = await browser.newPage(); 	
		//await page.goto(url); 
		const userAgent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36';
		await page.setUserAgent(userAgent);
		if(cookiesStr!==''){
			var cookiesObj =  JSON.parse(cookiesStr);//转换为json对象
			for(var i=0;i<cookiesObj.length;i++){
				var name = cookiesObj[i].name;  //取json中的值
				var value = cookiesObj[i].value;
				await page.setCookie(cookiesObj[i]);
			}
		}
		await page.goto(url, {timeout: 1800000}).then(() => {
			console.log('成功');
		 },()=> {
			console.log('超时');
			browser.close();
			res.send(null); 
		 });;
		
		await page.setViewport({width:1920, height:1080});
		if (jsonObjectStr!=='') {		
			var jsonObject = JSON.parse(jsonObjectStr);
			for(var k in jsonObject){
				var operatorJson = jsonObject[k];			
				var operatorName = operatorJson['operator'];
				console.log(operatorName);
				if (operatorName=='scroll'){		
					var scrollHeight = operatorJson['scrollHeight'];
					await page.evaluate(async () => {
						let h1 = document.body.scrollHeight;
						if (Number(scrollHeight) > h1) {
							window.scrollTo(0, h1);
						} else {						
							window.scrollTo(0, Number(scrollHeight));
						}				
					});      
				} else if (operatorName=='windowsSize') {
					var inputValue = operatorJson['inputValue'];
					var lengthArr=inputValue.split(";");
					if(lengthArr.length==2) {
						var new_width = Number(lengthArr[0]);
						var new_height = Number(lengthArr[1]);
						await page.setViewport({width:new_width, height:new_height});
					}
				} else {	
					var elementName = operatorJson['elementName'];
					var elementValue = operatorJson['elementValue'];
					var inputValue = operatorJson['inputValue'];
					if (elementName=='css'){
						await page.waitForSelector(elementValue, {timeout: 60000});
						if (operatorName=='click') {
							await page.click(elementValue);
						} else if (operatorName=='input'){							
							console.log(elementValue);
							console.log(inputValue);
							await page.type(elementValue, inputValue);
						}						
					} else if(elementName=='xpath'){
						await page.waitForXPath(elementValue, {timeout: 60000});
						var xpathElement = await page.$x(elementValue);
						if (operatorName=='click') {
							await xpathElement.click();
						} else if (operatorName=='input'){
							await elementHandle.type(inputValue);
						}				
					} 
				}				
				var wait = operatorJson['wait'];
				//console.log(wait);
				await page.waitFor(Number(wait));
			}
		}
		await page.waitFor(3000); 

		await page.screenshot({
				path: './test.png',
				fullPage :true
			 });		
		// await page.waitForNavigation({
		// 	waitUntil: 'load'
		 // });
		 // 页面渲染完毕后，开始截图
		if (areaSelectStr!=='') {
			var areaSelectJson = JSON.parse(areaSelectStr);
			var elementName = areaSelectJson['elementName'];
			var elementValue = areaSelectJson['elementValue'];
			console.log(elementValue);
			if (elementValue==''){
				await page.screenshot({
				path: './test.png',
				fullPage :true
			 });
			}else{
				var shotElement;
				if (elementName=='css'){
					await page.waitForSelector(elementValue, {timeout: 60000});
					shotElement = await page.$(elementValue);				
				} else if(elementName=='xpath'){
					await page.waitForXPath(elementValue, {timeout: 60000});
					shotElement = await page.$x(elementValue);									
				} 			
				await shotElement.screenshot({
					path: './test.png'
				}); 				
			}
			 
			 await fdfs.upload('./test.png').then(function(fileId) {
				var fileUrl = 'http://'+fdfs_host+'/'+fileId;
				res.set('_upUrl', fileUrl);
				console.log(fileUrl);				
			}).catch(function(err) {
				console.error(err);
			})
		}
		  		  
		var html;
		if(type=='page'){
			html = await page.content(); 
		} else {
			html = await page.cookies(); 		
		}
		browser.close();
		res.send(html); 
	})(); 
}) 

app.listen(5000, function () {
  console.log("程序启动")
}) 