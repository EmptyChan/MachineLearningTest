"use strict";
var port = 8088;
var server = require('webserver').create();

//服务端监听
server.listen(8088, function (request, response) {
	//传入的参数有待更改，目前为
	/*
{
	"url": "https://acg12.com/200340/",
	"operation": [{
	"click": {
	"dom": "a.p",
	"type": "id"
	}
	}, {
	"input": {
	"dom": "entity",
	"value": "password",
	"type": "class"
	}
	}
	],
	"result": "page & cookie"
	}

	 */
	//第一个参数为详情页，第二个为下载按钮的Dom
	//console.log(request.post);
	var url = request.post.url.toString();
	console.log(url);
	var operation = request.post.operation;
	console.log(operation);
	var result = request.post.result;
	console.log(result);
	var code = 0;
	var click_time = 0;
	var page = require('webpage').create();
	page.viewportSize = {
		width: 1920,
		height: 1080
	};
	//初始化headers
	page.onInitialized = function () {
		page.customHeaders = {
			"User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.120 Safari/537.36",
			"Referer": url
		};
	};
	page.settings.loadImages = true;
	page.customHeaders = {
		"User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.120 Safari/537.36",
		"Referer": url
	};
	response.headers = {
		'Cache': 'no-cache',
		'Content-Type': 'text/plain',
		'Connection': 'Keep-Alive',
		'Keep-Alive': 'timeout=20, max=100'
	};
	//根据Phantomjs的官网，这个回调在打开新标签页会触发
	page.onPageCreated = function (newPage) {
		console.log('A new child page was created! Its requested URL is not yet available, though.');
		newPage.onLoadFinished = function (status) {
			console.log('A child page is Loaded: ' + newPage.url);
			//newPage.render('newPage.png');
			response.write(newPage.url);
			response.statusCode = code;
			response.close();
		};
	};
	//嵌入的资源加载时会触发这个回调
	page.onResourceRequested = function (requestData, networkRequest) {
		//console.log('Request (#' + requestData.id + '): ' + JSON.stringify(requestData));
		//if (requestData.url.match(/.*google.*/g) != null) {
		//	console.log('Request (#' + requestData.id + '): ' + JSON.stringify(requestData));
		//	networkRequest.abort();
		//}
	};
	//资源加载失败时触发这个回调
	page.onResourceError = function (resourceError) {
		console.log('Unable to load resource (#' + resourceError.id + 'URL:' + resourceError.url + ')');
		console.log('Error code: ' + resourceError.errorCode + '. Description: ' + resourceError.errorString);
	};
	/*page.onNavigationRequested = function (url, type, willNavigate, main) {
	console.log('Trying to navigate to: ' + url);
	console.log('Caused by: ' + type);
	console.log('Will actually navigate: ' + willNavigate);
	console.log('Sent from the page\'s main frame: ' + main);
	};*/
	//让Phantomjs帮助我们去请求页面
	page.open(url, function (status) {
		console.log("----" + status + "----");
		if (status !== 'success') {
			code = 400;
			response.write('4XX');
			response.statusCode = code;
			response.close();
		} else {
			code = 200;
			// 通过在页面上执行脚本获取页面的渲染高度
			var bb = page.evaluate(function () {
					return document.getElementsByTagName('html')[0].getBoundingClientRect();
				});
			// 按照实际页面的高度，设定渲染的宽高
			page.clipRect = {
				top: bb.top,
				left: bb.left,
				width: bb.width,
				height: bb.height
			};

			window.setTimeout(function () {
				//执行JavaScript代码，类似于在浏览器Console中执行JavaScript
				click_time = page.evaluate(function (operation) {
						console.log("oooooooo");
						operation = '{"operation":' + operation + '}';
						var datas = JSON.parse(operation).operation;
						var click_time = 0;
						for (var i = 0; i < datas.length; i++) {
							var ele = datas[i];
							console.log("*******");
							var dom_el;
							if (ele.hasOwnProperty("click")) {
								click_time++;
								var doit = ele.click;
								console.log('dom::' + doit.dom);
								switch (doit.type) {
								case "id":
									dom_el = document.getElementById(doit.dom);
									break;
								case "class":
									dom_el = document.getElementsByClassName(doit.dom);
									break;
								case "tag":
									dom_el = document.getElementsByTagName(doit.dom);
									break;
								case "css":
									dom_el = document.querySelector(doit.dom);
									break;
								}
								if (dom_el != null) {
									dom_el.click();
									console.log(doit.dom + ' click!!!!');
								} else {
									console.log('error::' + doit.dom);
								}
							} else if (ele.hasOwnProperty("input")) {
								var doit = ele.input;
								switch (doit.type) {
								case "id":
									dom_el = document.getElementById(doit.dom);
									break;
								case "class":
									dom_el = document.getElementsByClassName(doit.dom);
									break;
								case "tag":
									dom_el = document.getElementsByTagName(doit.dom);
									break;
								case "css":
									dom_el = document.querySelector(doit.dom);
									break;
								}
								if (dom_el != null) {
									dom_el.value = doit.value;
									console.log(doit.dom + ' input>>>>' + doit.value);
								} else {
									console.log('error::' + doit.dom);
								}
							}
						}
						return click_time;
					}, operation);
			}, 50000);
			page.onUrlChanged = function (targetUrl) {
				console.log('New URL: ' + targetUrl);
			};
			//一级导航
			page.onNavigationRequested = function (url, type, willNavigate, main) {
				console.log('Trying to navigate to: ' + url);
				console.log('Caused by: ' + type);
				console.log('Will actually navigate: ' + willNavigate);
				console.log('Sent from the page\'s main frame: ' + main);
				if (click_time == 0) {
					page.onLoadFinished = function (status) {
						var return_str = result.split('&');
						var return_result = {
							"result": {}
						};
						for (var i = 0; i < return_str.length; i++) {
							var type = return_str[i].trim();
							switch (type) {
							case "page":
								return_result.result.page = page.content;
								break;
							case "cookie":
								var temp;
								var cookies = page.cookies;
								for (var i in cookies) {
									temp += cookies[i].name + '=' + cookies[i].value;
								}
								return_result.result.cookie = temp;
							}
						}
						page.render('page.png');
						response.write(JSON.stringify(return_result));
						response.statusCode = code;
						response.close();
					};
				}
				click_time--;
			};
		}
	});
	//根据Phantomjs的官网，这个回调主要应对执行evaluate函数内部的console.log输出，因为两个环境是隔离的。
	page.onConsoleMessage = function (msg, lineNum, sourceId) {
		console.log("$$$$$" + msg);
		console.log("$$$$$" + lineNum);
	};

	page.onError = function (msg, trace) {
		var msgStack = ['PHANTOM ERROR: ' + msg];
		if (trace && trace.length) {
			msgStack.push('TRACE:');
			trace.forEach(function (t) {
				msgStack.push(' -> ' + (t.file || t.sourceURL) + ': ' + t.line + (t.function  ? ' (in function ' + t.function  + ')': ''));
			});
		}
		console.log(msgStack.join('\n'));
		//phantom.exit(1);
	};
});
phantom.onError = function (msg, trace) {
	var msgStack = ['PHANTOM ERROR: ' + msg];
	if (trace && trace.length) {
		msgStack.push('TRACE:');
		trace.forEach(function (t) {
			msgStack.push(' -> ' + (t.file || t.sourceURL) + ': ' + t.line + (t.function  ? ' (in function ' + t.function  + ')': ''));
		});
	}
	console.log(msgStack.join('\n'));
	//phantom.exit(1);
};
