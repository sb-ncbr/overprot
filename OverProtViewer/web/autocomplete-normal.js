function autocomplete(inp, arr, showIfEmpty=false) {
    /*the autocomplete function takes two arguments,
    the text field element and an array of possible autocompleted values:*/

    function inputHandler(){
        let val = this.value;
        /*close any already open lists of autocompleted values*/
        closeAllLists();
        if (val == '' && !showIfEmpty) { return false; }
        currentFocus = -1;
        /*create a DIV element that will contain the items (values):*/
        let list = document.createElement("DIV");
        list.setAttribute("id", this.id + "autocomplete-list");
        list.setAttribute("class", "autocomplete-items");
        /*append the DIV element as a child of the autocomplete container:*/
        this.parentNode.appendChild(list);
        /*for each item in the array...*/
        for (let i = 0; i < arr.length; i++) {
            /*check if the item starts with the same letters as the text field value:*/
            if (arr[i].substr(0, val.length).toUpperCase() == val.toUpperCase()) {
                let item = createListItem(arr[i], val.length);
                list.appendChild(item);
            }
        }
    }

    function createListItem(text, nMatchChars){
        let b = document.createElement("DIV");
        /*make the matching letters bold:*/
        b.innerHTML = "<strong>" + text.substr(0, nMatchChars) + "</strong>";
        b.innerHTML += text.substr(nMatchChars);
        /*insert a input field that will hold the current array item's value:*/
        b.innerHTML += "<input type='hidden' value='" + text + "'>";
        b.value = text;
        /*execute a function when someone clicks on the item value (DIV element):*/
        b.addEventListener("click", selectHandler);
        return b;
    }

    function selectHandler(event, item){
        console.log(event, item, this);
        item = item ?? this;
        /*insert the value for the autocomplete text field:*/
        inp.value = item.getElementsByTagName("input")[0].value;
        /*close the list of autocompleted values,
        (or any other open lists of autocompleted values:*/
        closeAllLists();
    }

    var currentFocus;
    /*execute a function when someone writes in the text field:*/
    inp.addEventListener("input", inputHandler);
    inp.addEventListener("click", inputHandler);

    /*execute a function presses a key on the keyboard:*/
    inp.addEventListener("keydown", function(e) {
        var x = document.getElementById(this.id + "autocomplete-list");
        if (x) x = x.getElementsByTagName("div");
        if (e.keyCode == 40) {
            /*If the arrow DOWN key is pressed,
            increase the currentFocus variable:*/
            currentFocus++;
            /*and and make the current item more visible:*/
            addActive(x);
        } else if (e.keyCode == 38) { //up
            /*If the arrow UP key is pressed,
            decrease the currentFocus variable:*/
            currentFocus--;
            /*and and make the current item more visible:*/
            addActive(x);
        } else if (e.keyCode == 13) {
            /*If the ENTER key is pressed, prevent the form from being submitted,*/
            // e.preventDefault();
            if (currentFocus > -1) {
                e.preventDefault();
                /*and simulate a click on the "active" item:*/
                if (x) { 
                    selectHandler(e, x[currentFocus])
                    // x[currentFocus].click();
                }
                inputHandler();
            }
        }
    });

    function addActive(x) {
        /*a function to classify an item as "active":*/
        if (!x) return false;
        /*start by removing the "active" class on all items:*/
        removeActive(x);
        if (currentFocus >= x.length) currentFocus = 0;
        if (currentFocus < 0) currentFocus = (x.length - 1);
        /*add class "autocomplete-active":*/
        x[currentFocus].classList.add("autocomplete-active");
    }

    function removeActive(x) {
        /*a function to remove the "active" class from all autocomplete items:*/
        for (var i = 0; i < x.length; i++) {
        x[i].classList.remove("autocomplete-active");
        }
    }

    function closeAllLists(elmnt) {
        /*close all autocomplete lists in the document,
        except the one passed as an argument:*/
        var x = document.getElementsByClassName("autocomplete-items");
        for (var i = 0; i < x.length; i++) {
            if (elmnt != x[i] && elmnt != inp) {
                x[i].parentNode.removeChild(x[i]);
            }
        }
    }

    /*execute a function when someone clicks in the document:*/
    document.addEventListener("click", function (e) {
        closeAllLists(e.target);
    });
}