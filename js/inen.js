function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function expand_fields() {
	await sleep(1000);
	var aa= document.getElementsByClassName("ui-icon ui-icon-triangle-1-e");
	for (var i = 0; i < aa.length; i++) {
		aa[i].click()
	}
	console.log('fields left:' + aa.length)
	if (aa.length > 0) {
		expand_fields()
	}
}

async function open_fields() {
	console.log('Openning cities...');
	document.getElementById('formIzquierda:idAddAmbito').click();
	for (var i = 0; i < 25; i++) {
		await sleep(1000);
		var x = document.getElementById('formIzquierda:j_idt29');
		x.children[1].selected = true;
		x.onchange();
		await sleep(1000);
		document.getElementById('formIzquierda:idBtnFooter').click();
	}
	console.log('Expanding fields...');
	expand_fields();
	console.log('Done.')
}

open_fields();


var aa= document.getElementsByClassName("ui-chkbox-box ui-widget");
for (var i = 450; i < aa.length; i++) {
	aa[i].click();
}
console.log(i);
