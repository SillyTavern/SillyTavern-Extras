const MODULE_NAME = 'expressions';
const SETTINGS_KEY = 'extensions_memory_settings';
const UPDATE_INTERVAL = 1000;

let expressionsList = null;
let lastCharacter = undefined;
let lastMessage = null;
let inApiCall = false;

const getContext = function () {
    return window['TavernAI'].getContext();
}

const getApiUrl = function () {
    return localStorage.getItem('extensions_url');
}

async function moduleWorker() {
    function getLastCharacterMessage() {
        const reversedChat = context.chat.slice().reverse();

        for (let mes of reversedChat) {
            if (mes.is_user || mes.is_system) {
                continue;
            }

            return mes.mes;
        }

        return '';
    }

    const context = getContext();

    // group chats and non-characters not supported
    if (context.groupId || !context.characterId) {
        removeExpression();
        return;
    }

    // character changed
    if (lastCharacter !== context.characterId) {
        removeExpression();
        validateImages();
    }

    // check if last message changed
    const currentLastMessage = getLastCharacterMessage();
    if (lastCharacter === context.characterId && lastMessage === currentLastMessage) {
        return;
    }

    // API is busy
    if (inApiCall) {
        return;
    }

    try {
        inApiCall = true;
        const url = new URL(getApiUrl());
        url.pathname = '/api/classify';

        const apiResult = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Bypass-Tunnel-Reminder': 'bypass',
            },
            body: JSON.stringify({ text: currentLastMessage })
        });

        if (apiResult.ok) {
            const data = await apiResult.json();
            const expression = data.classification[0].label;
            setExpression(context.name2, expression);
        }

    }
    catch (error) {
        console.log(error);
    }
    finally {
        inApiCall = false;
        lastCharacter = context.characterId;
        lastMessage = currentLastMessage;
    }
}

function removeExpression() {
    $('div.expression').css('background-image', 'unset');
    $('.expression_settings').hide();
}

async function validateImages() {
    const context = getContext();
    $('.expression_settings').show();
    $('#image_list').empty();

    if (!context.characterId) {
        return;
    }

    const IMAGE_LIST = (await getExpressionsList()).map(x => `${x}.png`);
    IMAGE_LIST.forEach((item) => {
        const image = document.createElement('img');
        image.src = `/characters/${context.name2}/${item}`;
        image.classList.add('debug-image');
        image.width = '0px';
        image.height = '0px';
        image.onload = function() {
            $('#image_list').append(`<li class="success">${item} - OK</li>`);
        }
        image.onerror = function() {
            $('#image_list').append(`<li class="failure">${item} - Missing</li>`);
        }
        $('#image_list').prepend(image);
    });
}

async function getExpressionsList() {
    if (Array.isArray(expressionsList)) {
        return expressionsList;
    }

    const url = new URL(getApiUrl());
    url.pathname = '/api/classify/labels';

    try {
        const apiResult = await fetch(url, {
            method: 'GET',
            headers: { 'Bypass-Tunnel-Reminder': 'bypass' },
        });
    
        if (apiResult.ok) {
            const data = await apiResult.json();
            expressionsList = data.labels;
            return expressionsList;
        }
    }
    catch (error) {
        console.log(error);
        return [];
    }
}

function setExpression(character, expression) {
    const imgUrl = `url('/characters/${character}/${expression}.png')`;
    $('div.expression').css('background-image', imgUrl);
}

(function () {
    function addExpressionImage() {
        const html = `<div class="expression"></div>`
        $('body').append(html);
    }
    function addSettings() {
        const html = `
        <div class="expression_settings">
            <h4>Expression images</h4>
            <ul id="image_list"></ul>
            <p><b>Hint:</b> <i>Put images into the <tt>public/characters/Name</tt>
            folder of TavernAI, where Name is the name of the character</i></p>
        </div>
        `;
        $('#extensions_settings').append(html);
        $('.expression_settings').hide();
    }

    addExpressionImage();
    addSettings();
    setInterval(moduleWorker, UPDATE_INTERVAL);
})();