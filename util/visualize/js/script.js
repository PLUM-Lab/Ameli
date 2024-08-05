var coll = document.getElementsByClassName("collapsible");
var i;

for (i = 0; i < coll.length; i++) {
  coll[i].addEventListener("click", function() {

    var content = this.nextElementSibling;
    if (content.style.display === "block") {
      content.style.display = "none";
    } else {
      content.style.display = "block";
    }
  });
}


var expand_all_button = document.getElementsByClassName("expand_all");
var j;

for (j = 0; j < expand_all_button.length; j++) {
  expand_all_button[j].addEventListener("click", function() {
   
  
    for (j= 0;j < coll.length; j++) {
      var content = coll[j].nextElementSibling;
      if (this.innerText == 'Expand All') { 
         
        content.style.display = "block";
      } else {
        content.style.display = "none";
      }
    }
    if (this.innerText == 'Expand All') { 
      this.innerHTML = 'Collapse All';
 
    } else { 
      this.innerHTML = 'Expand All';
    
    }
 
 
  });
}

// var CollapsibleLists =
//   new function(){
//     // Post-order traversal of the collapsible list(s) 
//     // if collapse is true, then all list items implode, else they explode.
//     this.collapse = function(collapse){
//       for (i = 0; i < coll.length; i++) {
//           var content = coll[i].nextElementSibling;
//           if (collapse) {
//             content.style.display = "none";
//           } else {
//             content.style.display = "block";
//           }
//       }
//     };
// }();

// function listExpansion() {
//   var element = document.getElementById('listHeader');
//   if (element.innerText == 'Expand All') { 
//     element.innerHTML = 'Collapse All';
//     CollapsibleLists.collapse(false); 
//   } else { 
//     element.innerHTML = 'Expand All';
//     CollapsibleLists.collapse(true);
//   }
// }


//image
window.onload = () => {
  // (A) GET ALL IMAGES
  let all = document.getElementsByClassName("zoomE");
  
  // (B) CLICK TO GO FULLSCREEN
  if (all.length>0) { for (let i of all) {
    i.onclick = () => {
      // (B1) EXIT FULLSCREEN
      if (document.fullscreenElement != null || document.webkitFullscreenElement != null) {
        if (document.exitFullscreen) { document.exitFullscreen(); }
        else { document.webkitCancelFullScreen(); }
      }
  
      // (B2) ENTER FULLSCREEN
      else {
        if (i.requestFullscreen) { i.requestFullscreen(); }
        else { i.webkitRequestFullScreen(); }
      }
    };
  }}
};
