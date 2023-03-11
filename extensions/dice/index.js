// Borrowed from the Droll library by thebinarypenguin
// https://github.com/thebinarypenguin/droll
// Licensed under MIT license
var droll = {};

// Define a "class" to represent a formula
function DrollFormula() {
    this.numDice = 0;
    this.numSides = 0;
    this.modifier = 0;

    this.minResult = 0;
    this.maxResult = 0;
    this.avgResult = 0;
}

// Define a "class" to represent the results of the roll
function DrollResult() {
    this.rolls = [];
    this.modifier = 0;
    this.total = 0;
}

/**
 * Returns a string representation of the roll result
 */
DrollResult.prototype.toString = function () {
    if (this.rolls.length === 1 && this.modifier === 0) {
        return this.rolls[0] + '';
    }

    if (this.rolls.length > 1 && this.modifier === 0) {
        return this.rolls.join(' + ') + ' = ' + this.total;
    }

    if (this.rolls.length === 1 && this.modifier > 0) {
        return this.rolls[0] + ' + ' + this.modifier + ' = ' + this.total;
    }

    if (this.rolls.length > 1 && this.modifier > 0) {
        return this.rolls.join(' + ') + ' + ' + this.modifier + ' = ' + this.total;
    }

    if (this.rolls.length === 1 && this.modifier < 0) {
        return this.rolls[0] + ' - ' + Math.abs(this.modifier) + ' = ' + this.total;
    }

    if (this.rolls.length > 1 && this.modifier < 0) {
        return this.rolls.join(' + ') + ' - ' + Math.abs(this.modifier) + ' = ' + this.total;
    }
};

/**
 * Parse the formula into its component pieces.
 * Returns a DrollFormula object on success or false on failure.
 */
droll.parse = function (formula) {
    var pieces = null;
    var result = new DrollFormula();

    pieces = formula.match(/^([1-9]\d*)?d([1-9]\d*)([+-]\d+)?$/i);
    if (!pieces) { return false; }

    result.numDice = (pieces[1] - 0) || 1;
    result.numSides = (pieces[2] - 0);
    result.modifier = (pieces[3] - 0) || 0;

    result.minResult = (result.numDice * 1) + result.modifier;
    result.maxResult = (result.numDice * result.numSides) + result.modifier;
    result.avgResult = (result.maxResult + result.minResult) / 2;

    return result;
};

/**
 * Test the validity of the formula.
 * Returns true on success or false on failure.
 */
droll.validate = function (formula) {
    return (droll.parse(formula)) ? true : false;
};

/**
 * Roll the dice defined by the formula.
 * Returns a DrollResult object on success or false on failure.
 */
droll.roll = function (formula) {
    var pieces = null;
    var result = new DrollResult();

    pieces = droll.parse(formula);
    if (!pieces) { return false; }

    for (var a = 0; a < pieces.numDice; a++) {
        result.rolls[a] = (1 + Math.floor(Math.random() * pieces.numSides));
    }

    result.modifier = pieces.modifier;

    for (var b = 0; b < result.rolls.length; b++) {
        result.total += result.rolls[b];
    }
    result.total += result.modifier;

    return result;
};

// END OF DROLL CODE

const MODULE_NAME = 'dice';
const UPDATE_INTERVAL = 1000;

const getContext = () => window['TavernAI'].getContext();
const getApiUrl = () => localStorage.getItem('extensions_url');

let popper;

async function urlContentToDataUri(url, params) {
    const response = await fetch(url, params);
    const blob = await response.blob();
    return await new Promise(callback => {
        let reader = new FileReader();
        reader.onload = function () { callback(this.result); };
        reader.readAsDataURL(blob);
    });
}

async function setDiceIcon() {
    try {
        const sendButton = document.getElementById('roll_dice');
        const imgUrl = new URL(getApiUrl());
        imgUrl.pathname = `/api/asset/${MODULE_NAME}/dice-solid.svg`;
        const dataUri = await urlContentToDataUri(imgUrl.toString(), { method: 'GET', headers: { 'Bypass-Tunnel-Reminder': 'bypass' } });
        sendButton.style.backgroundImage = `url(${dataUri})`;
        sendButton.classList.remove('spin');
    }
    catch (error) {
        console.log(error);
    }
}

function doDiceRoll() {
    const value = $(this).data('value');
    const isValid = droll.validate(value);

    if (isValid) {
        const result = droll.roll(value);
        const context = getContext();
        context.sendSystemMessage('generic', `${context.name1} rolls the ${value}. The result is: ${result.total}`);
    }
}

function show() {
    document.getElementById('dice_dropdown').setAttribute('data-show', '');
}

function hide() {
    document.getElementById('dice_dropdown').removeAttribute('data-show');
}

function addDiceRollButton() {
    const buttonHtml = `
        <input  id="roll_dice" type="button" />
        <div id="dice_dropdown">
            <ul class="list-group">
                <li class="list-group-item" data-value="d4">d4</li>
                <li class="list-group-item" data-value="d6">d6</li>
                <li class="list-group-item" data-value="d8">d8</li>
                <li class="list-group-item" data-value="d10">d10</li>
                <li class="list-group-item" data-value="d12">d12</li>
                <li class="list-group-item" data-value="d20">d20</li>
                <li class="list-group-item" data-value="d100">d100</li>
            </ul>
        </div>
        `;

    $('#send_but_sheld').prepend(buttonHtml);
    $('#dice_dropdown li').on('click', doDiceRoll);
    const button = $('#roll_dice');
    const dropdown = $('#dice_dropdown');
    dropdown.hide();
    button.hide();

    popper = Popper.createPopper(button.get(0), dropdown.get(0), {
        placement: 'top-start',
    });

    $(document).on('click touchend', function (e) {
        const target = $(e.target);
        if (target.is(dropdown)) return;
        if (target.is(button) && !dropdown.is(":visible")) {
            e.preventDefault();

            dropdown.show();
            popper.update();
        } else {
            dropdown.hide();
        }
    });
}

function patchSendForm() {
    const columns = $('#send_form').css('grid-template-columns').split(' ');
    columns[columns.length - 1] = `${parseInt(columns[columns.length - 1]) + 40}px`;
    columns[1] = 'auto';
    $('#send_form').css('grid-template-columns', columns.join(' '));
}


async function moduleWorker() {
    const context = getContext();

    context.onlineStatus === 'no_connection'
        ? $('#roll_dice').hide(200)
        : $('#roll_dice').show(200);
}

(function () {
    addDiceRollButton();
    patchSendForm();
    setDiceIcon();
    setInterval(moduleWorker, UPDATE_INTERVAL);
})();