// const e = require("express");

function autocomplete(inputElement, options, maxItems=10, openIfEmpty=false, selectAllOnFocus=true) {
    /*the autocomplete function takes two arguments,
    the text field element and an array of possible autocompleted values:*/

    if (inputElement.getAttribute('autocomplete-set') == 'true'){
        console.error('Repeated initialization of autocomplete on ', inputElement);
        return;
    }
    inputElement.setAttribute('autocomplete-set', 'true');
    inputElement.setAttribute('autocomplete', 'off');

    let autocompleteDiv = inputElement.parentNode;
    let currentFocus;

    function inputHandler(){
        let val = inputElement.value;
        /*close any already open lists of autocompleted values*/
        closeAllLists();
        if (val == '' && !openIfEmpty) { return false; } // DEBUG
        currentFocus = -1;
        /*create a DIV element that will contain the items (values):*/
        let list = document.createElement("DIV");
        list.setAttribute("id", inputElement.id + "autocomplete-list");
        list.setAttribute("class", "autocomplete-items");
        /*append the DIV element as a child of the autocomplete container:*/
        autocompleteDiv.appendChild(list);
        /*for each item in the array...*/
        let matches = getMatches(val, options, maxNumber=maxItems);
        matches.forEach(opt => list.appendChild(createListItem(opt, val.length)));
    }

    function getMatches(currentValue, options, maxNumber=20){
        let result = [];
        for (let i = 0; i < options.length; i++){
            let [value, hint, children] = option = options[i];
            if (startswith(currentValue, value) && children != undefined) {
                return getMatches(currentValue, children, maxNumber=maxNumber);
            }
            if (startswith(value, currentValue)) {
                result.push(option);
            }
            if (result.length >= maxNumber){
                break;
            }
        }
        return result;
    }
    function startswith(text, prefix){
        return text.substr(0, prefix.length).toLowerCase() == prefix.toLowerCase();
    }

    function createListItem(option, nMatchChars){
        let [value, hint, children] = option;
        let item = document.createElement("DIV");
        item.classList.add('autocomplete-item');
        item.setAttribute('value', value);
        item.setAttribute('keep-open', children != undefined);
        item.setAttribute('do-submit', children == undefined);
        let matched = value.substr(0, nMatchChars);
        let completed = value.substr(nMatchChars);
        item.innerHTML = `<span class='matched'>${matched}</span>`
                        + `<span class='completed'>${completed}</span>`
                        + `<span class='hint'> &nbsp; ${hint??''}</span>`;  //&#x22EF;
        item.value = value;
        item.addEventListener("click", (event) => selectHandler(event, item));
        return item;
    }

    function selectHandler(event, item){
        event.stopPropagation();
        event.preventDefault();
        item = item ?? this;
        /*insert the value for the autocomplete text field:*/
        inputElement.value = item.getAttribute('value');  // item.getElementsByTagName("input")[0].value;
        /*close the list of autocompleted values,(or any other open lists of autocompleted values:*/
        closeAllLists();
        if (item.getAttribute('keep-open') == 'true'){
            inputHandler();
        }
        if (item.getAttribute('do-submit') == 'true'){
            inputElement.form.submit();
        }

    }

    function selectAllInputText(event){
        let elem = event.target;
        elem.setSelectionRange(0, elem.value.length);
    }

    /*execute a function when someone writes in the text field:*/
    inputElement.addEventListener("input", inputHandler);
    inputElement.addEventListener("click", inputHandler);
    if (selectAllOnFocus) {
        inputElement.addEventListener("focus", selectAllInputText);
    }

    function keydownHandler(e){
        e.stopPropagation();
        var items = document.getElementById(this.id + "autocomplete-list");
        if (!items) {
            return;
        }
        items = items.getElementsByTagName("div");
        // x = x.getElementsByClassName('autocomplete-items');
        if (e.keyCode == 40) {
            e.preventDefault();
            /*If the arrow DOWN key is pressed,
            increase the currentFocus variable:*/
            currentFocus++;
            if (currentFocus >= items.length) currentFocus = 0;
            /*and and make the current item more visible:*/
            addActive(items);
        } else if (e.keyCode == 38) { //up
            e.preventDefault();
            /*If the arrow UP key is pressed,
            decrease the currentFocus variable:*/
            currentFocus--;
            if (currentFocus < 0) currentFocus = (items.length - 1);
            /*and and make the current item more visible:*/
            addActive(items);
        } else if (e.keyCode == 13) { //enter
            if (currentFocus >= 0) {
                e.preventDefault();
                /*and simulate a click on the "active" item:*/
                if (items[currentFocus]) {
                    selectHandler(e, items[currentFocus]);
                    // items[currentFocus].click();
                }
            }
        }
    }

    /*execute a function presses a key on the keyboard:*/
    inputElement.addEventListener("keydown", keydownHandler);

    function addActive(x) {
        /*a function to classify an item as "active":*/
        if (!x) return false;
        /*start by removing the "active" class on all items:*/
        removeActive(x);
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
        if (elmnt == inputElement) {
            return;
        }
        var listElems = autocompleteDiv.getElementsByClassName('autocomplete-items');
        for (let listElem of listElems){
            if (elmnt != listElem){
                autocompleteDiv.removeChild(listElem);
            }
        }
        currentFocus = -1;
    }
    
    /*execute a function when someone clicks in the document:*/
    document.addEventListener("click", function (e) {
        closeAllLists(e.target);
    });
}

function getFamilyName(familyAutoCompleteOpts, familyId) { // -> string|null
    if (familyAutoCompleteOpts==undefined || familyAutoCompleteOpts==null){
        return null;
    }
    for (let i = 0; i < familyAutoCompleteOpts.length; i++){
        let [id, name, children] = familyAutoCompleteOpts[i];
        if (familyId == id){
            return name ?? null;
        }
        if (familyId.substr(0, id.length) == id){
            return getFamilyName(children, familyId);
        }
    }
}

function cathFamiliesAutocompleteOptions(familyIdsNames){
    function optAdd(optsDict, key_path, value){
        let key = key_path.shift();
        if (key_path.length == 0){
            let opt = {value: key, hint: value};
            optsDict[key] = opt;
        } else {
            if (optsDict[key] == undefined){
                optsDict[key] = {value: key, hint: ''};
            }
            if (optsDict[key].children == undefined){
                optsDict[key].children = {};
            }
            key_path[0] = key + '.' + key_path[0];
            optAdd(optsDict[key].children, key_path, value);
        }
    }
    function optsDictToOpts(optsDict, depth){
        let result = Object.values(optsDict);
        for (let i = 0; i < result.length; i++){
            let item = result[i];
            if (depth > 1){
                item.value = item.value + '.';
            }
            if (item.children != undefined){
                item.children = optsDictToOpts(item.children, depth-1);
            }
        }
        return result;
    }
    let result = {};
    familyIdsNames.split('\n').forEach(family => {
        if (family != ''){
            let id_names = family.split(' ');
            let id = id_names.shift();
            let name = id_names.join(' ');
            let parts = id.split('.');
            optAdd(result, parts, name);
        }
    });

    return optsDictToOpts(result, 4);
}

// function initFamilyInputAutocompleteAndFamilyName(optionsUrl, familyId, familyInputElement=null, familyNameClass=null){
//     // console.log('INIT:', optionsUrl, familyId, familyInputElement, familyNameClass);
//     fetch(optionsUrl)
//         .then(response => response.text())
//         .then(text => {
//             let opts = JSON.parse(text);
//             if (familyInputElement){
//                 let familyNameElems = document.getElementsByClassName(familyNameClass);
//                 let familyName = getFamilyName(opts, familyId);
//                 for (let i = 0; i < familyNameElems.length; i++){
//                     elem = familyNameElems[i];
//                     familyNameElems[i].innerHTML = familyName;
//                 }
//             }
//             if (familyInputElement){
//                 let elem = document.getElementById(familyInputElement);
//                 if (elem){
//                     autocomplete(document.getElementById(familyInputElement), opts, maxItems=10, openIfEmpty=true, selectAllOnFocus=true);
//                 }
//             }
//         });
// }
function initFamilyInputAutocompleteAndFamilyName(optionsUrl, familyId, familyInputElementSelector, familyNameSelector){
    console.log('initFamilyInputAutocompleteAndFamilyName:', optionsUrl, familyId, familyInputElementSelector, familyNameSelector);
    fetch(optionsUrl)
        .then(response => response.text())
        .then(text => {
            let opts = JSON.parse(text);
            if (familyNameSelector){
                let familyNameElems = document.querySelectorAll(familyNameSelector);
                let familyName = getFamilyName(opts, familyId);
                for (let elem of familyNameElems){
                    elem.innerHTML = familyName;
                }
            }
            if (familyInputElementSelector){
                let elems = document.querySelectorAll(familyInputElementSelector);
                for (let elem of elems){
                    autocomplete(elem, opts, maxItems=10, openIfEmpty=true, selectAllOnFocus=true);
                }
            }
        });
}