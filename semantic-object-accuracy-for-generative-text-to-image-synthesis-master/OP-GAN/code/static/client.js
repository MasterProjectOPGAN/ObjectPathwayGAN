var el = x => document.getElementById(x);

function showPicker() {
  el("file-input").click();
}

function showPicked(input) {
  var e = el("myDropdown")
  var text = e.options[e.selectedIndex].text;
  console.log(text)
//el("upload-label").innerHTML = input.files[0].name;
  //var reader = new FileReader();
  //reader.onload = function(e) {
  //  el("image-picked").src = e.target.result;
  //  el("image-picked").className = "";
  //};
  //reader.readAsDataURL(input.files[0]);
}

function analyze() {
  //var uploadFiles = el("file-input").files;
  //if (uploadFiles.length !== 1) alert("Please select a file to analyze!");
  console.log('Inside analyze');
  var inputStr = el("text1").value
  console.log('analyze', inputStr)
  el("analyze-button").innerHTML = "Generating...";
  var xhr = new XMLHttpRequest();
  var loc = window.location;
  xhr.open("POST", `${loc.protocol}//${loc.hostname}:${loc.port}/analyze`,
    true);
  xhr.onerror = function() {
    alert(xhr.responseText);
  };
  xhr.onload = function(e) {
    if (this.readyState === 4) {
      var response = JSON.parse(e.target.responseText);
      el("image-path0").innerHTML = `${response["caption"]}`;
      console.log(response);
	    el("image-path1").innerHTML = "Image generated from 15th checkpoint model";
      document.querySelector("#image-path-new1").src = `${response["res"]}`;
	    el("image-path2").innerHTML = "Image generated from 20th checkpoint model";
      document.querySelector("#image-path-new2").src = `${response["resul"]}`;
	    el("image-path").innerHTML = "Image generated from 25th checkpoint model";
      document.querySelector("#image-path-new").src = `${response["result"]}`;
    }
    el("analyze-button").innerHTML = "Generate Image";
  };

  var fileData = new FormData();
  fileData.append("file", inputStr);
  xhr.send(fileData);
}

