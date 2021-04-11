var el = x => document.getElementById(x);

function showPicker() {
  el("file-input").click();
}

function showPicked(input) {
  el("upload-label").innerHTML = input.files[0].name;
  var reader = new FileReader();
  reader.onload = function(e) {
    el("image-picked").src = e.target.result;
    el("image-picked").className = "";
  };
  reader.readAsDataURL(input.files[0]);
}

function analyze() {
  //var uploadFiles = el("file-input").files;
  //if (uploadFiles.length !== 1) alert("Please select a file to analyze!");
  console.log('Inside analyze');
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
      //el("result-label").innerHTML = `Result = ${response["result"]}`;
      el("image-path").innerHTML = `${response["result"]}`;
      console.log(response);
      //var urlCreator = window.URL || window.webkitURL;
      //var imageUrl = urlCreator.createObjectURL(this.response);
      document.querySelector("#image-path-new").src = `${response["result"]}`;

    }
    el("analyze-button").innerHTML = "Generate Image";
  };

  var fileData = new FormData();
  //fileData.append("file", uploadFiles[0]);
  xhr.send(fileData);
}
