const MODULE_NAME = 'expressions';
const SETTINGS_KEY = 'extensions_memory_settings';
const UPDATE_INTERVAL = 1000;

let lastCharacter = null;
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

    addExpressionImage();
    setInterval(moduleWorker, UPDATE_INTERVAL);
})();