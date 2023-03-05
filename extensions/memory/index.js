const MODULE_NAME = 'memory';
const SETTINGS_KEY = 'extensions_memory_settings';
const UPDATE_INTERVAL = 1000;

let lastCharacterId = null;
let lastGroupId = null;
let lastChatId = null;
let lastMessageHash = null;
let lastMessageId = null;
let inApiCall = false;

function getStringHash(str, seed = 0) {
    let h1 = 0xdeadbeef ^ seed,
        h2 = 0x41c6ce57 ^ seed;
    for (let i = 0, ch; i < str.length; i++) {
        ch = str.charCodeAt(i);
        h1 = Math.imul(h1 ^ ch, 2654435761);
        h2 = Math.imul(h2 ^ ch, 1597334677);
    }

    h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
    h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);

    return 4294967296 * (2097151 & h2) + (h1 >>> 0);
};

const getContext = () => window['TavernAI'].getContext();
const getApiUrl = () => localStorage.getItem('extensions_url');
const formatMemoryValue = (value) => `[Context: "${value}"]`;

const defaultSettings = {
    minLongMemory: 16,
    maxLongMemory: 512,
    longMemoryLength: 128,
    shortMemoryLength: 512,
    minShortMemory: 128,
    maxShortMemory: 2048,
    shortMemoryStep: 16,
    longMemoryStep: 8,
};

const settings = {
    shortMemoryLength: defaultSettings.shortMemoryLength,
    longMemoryLength: defaultSettings.longMemoryLength,
}

function saveSettings() {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

function loadSettings() {
    const savedSettings = JSON.parse(localStorage.getItem(SETTINGS_KEY));

    if (savedSettings) {
        Object.assign(settings, savedSettings);
        $('#memory_long_length').val(settings.longMemoryLength).trigger('input');
        $('#memory_short_length').val(settings.shortMemoryLength).trigger('input');
    }
}

function onMemoryShortInput() {
    const value = $(this).val();
    settings.shortMemoryLength = Number(value);
    $('#memory_short_length_tokens').text(value);
    saveSettings();
}

function onMemoryLongInput() {
    const value = $(this).val();
    settings.longMemoryLength = Number(value);
    $('#memory_long_length_tokens').text(value);
    saveSettings();
}

function saveLastValues() {
    const context = getContext();
    lastGroupId = context.groupId;
    lastCharacterId = context.characterId;
    lastChatId = context.chatId;
    lastMessageId = context.chat?.length ?? null;
    lastMessageHash = getStringHash((context.chat.length && context.chat[context.chat.length - 1]['mes']) ?? '');
}

function getLatestMemoryFromChat(chat) {
    if (!Array.isArray(chat) || !chat.length) {
        return '';
    }

    const reversedChat = chat.slice().reverse();
    for (let mes of reversedChat) {
        if (mes.extra && mes.extra.memory) {
            return mes.extra.memory;
        }
    }

    return '';
}

async function moduleWorker() {
    const context = getContext();
    const chat = context.chat;

    // no characters or group selected 
    if (!context.groupId && !context.characterId) {
        return;
    }

    // Chat/character/group changed
    if ((context.groupId && lastGroupId !== context.groupId) || (context.characterId !== lastCharacterId) || (context.chatId !== lastChatId)) {
        const latestMemory = getLatestMemoryFromChat(chat);
        setMemoryContext(latestMemory, false);
        saveLastValues();
        return;
    }

    // Currently summarizing - skip
    if (inApiCall) {
        return;
    }

    // No new messages - do nothing
    if (lastMessageId === chat.length && getStringHash(chat[chat.length - 1].mes) === lastMessageHash) {
        return;
    }

    // Messages has been deleted - rewrite the context with the latest available memory
    if (chat.length < lastMessageId) {
        const latestMemory = getLatestMemoryFromChat(chat);
        setMemoryContext(latestMemory, false);
    }

    // Message has been edited / regenerated - delete the saved memory
    if (chat.length
        && chat[chat.length - 1].extra
        && chat[chat.length - 1].extra.memory
        && lastMessageId === chat.length
        && getStringHash(chat[chat.length - 1].mes) !== lastMessageHash) {
        delete chat[chat.length - 1].extra.memory;
    }

    try {
        await summarizeChat(context);
    }
    catch (error) {
        console.log(error);
    }
    finally {
        saveLastValues();
    }
}

async function summarizeChat(context) {
    function getMemoryString() {
        return (longMemory + '\n\n' + memoryBuffer.slice().reverse().join('\n\n')).trim();
    }

    const chat = context.chat;
    const longMemory = getLatestMemoryFromChat(chat);
    const reversedChat = chat.slice().reverse();
    let memoryBuffer = [];

    for (let mes of reversedChat) {
        // we reached the point of latest memory
        if (longMemory && mes.extra && mes.extra.memory == longMemory) {
            break;
        }

        // don't care about system
        if (mes.is_system) {
            continue;
        }

        // determine the sender's name
        const name = mes.is_user ? (context.name1 ?? 'You') : (mes.force_avatar ? mes.name : context.name2);
        const entry = `${name}:\n${mes['mes']}`;
        memoryBuffer.push(entry);

        // check if token limit was reached
        if (context.encode(getMemoryString()).length >= settings.shortMemoryLength) {
            break;
        }
    }

    const resultingString = getMemoryString();

    if (context.encode(resultingString).length < settings.shortMemoryLength) {
        return;
    }

    // perform the summarization API call
    try {
        inApiCall = true;
        const url = new URL(getApiUrl());
        url.pathname = '/api/summarize';

        const apiResult = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Bypass-Tunnel-Reminder': 'bypass',
            },
            body: JSON.stringify({
                text: resultingString,
                params: {
                    min_length: settings.longMemoryLength,
                    max_length: settings.longMemoryLength * 1.25
                }
            })
        });

        if (apiResult.ok) {
            const data = await apiResult.json();
            const summary = data.summary;

            const newContext = getContext();

            // something changed during summarization request
            if (newContext.groupId !== context.groupId || newContext.chatId !== context.chatId || (!newContext.groupId && (newContext.characterId !== context.characterId))) {
                return;
            }

            setMemoryContext(summary, true);
        }
    }
    catch (error) {
        console.log(error);
    }
    finally {
        inApiCall = false;
    }
}

function onMemoryRestoreClick() {
    const context = getContext();
    const content = $('#memory_contents').val();
    const reversedChat = context.chat.slice().reverse();

    for (let mes of reversedChat) {
        if (mes.extra && mes.extra.memory == content) {
            delete mes.extra.memory;
            break;
        }
    }

    const newContent = getLatestMemoryFromChat(context.chat);
    setMemoryContext(newContent, false);
}

function onMemoryContentInput() {
    const value = $(this).val();
    setMemoryContext(value, true);
}

function setMemoryContext(value, saveToMessage) {
    const context = getContext();
    context.setExtensionPrompt(formatMemoryValue(value));
    $('#memory_contents').val(value);

    if (saveToMessage && context.chat.length) {
        const mes = context.chat[context.chat.length - 1];
        !mes.extra && (mes.extra = {});
        mes.extra.memory = value;
    }
}

$(document).ready(function () {
    function addExtensionControls() {
        const settingsHtml = `
        <h4>Memory</h4>
        <div id="memory_settings">
            <label for="memory_contents">Memory contents</label>
            <textarea id="memory_contents" class="text_pole" rows="4" placeholder="Context will be generated here...">
            </textarea>
            <label for="memory_short_length">Memory summarization [short-term] length (<span id="memory_short_length_tokens"></span> tokens)</label>
            <input id="memory_short_length" type="range" value="${defaultSettings.shortMemoryLength}" min="${defaultSettings.minShortMemory}" max="${defaultSettings.maxShortMemory}" step="${defaultSettings.shortMemoryStep}" />
            <label for="memory_long_length">Memory context [long-term] length (<span id="memory_long_length_tokens"></span> tokens)</label>
            <input id="memory_long_length" type="range" value="${defaultSettings.longMemoryLength}" min="${defaultSettings.minLongMemory}" max="${defaultSettings.maxLongMemory}" step="${defaultSettings.longMemoryStep}" />
            <input id="memory_restore" type="button" value="Restore previous state" />
        </div>
        `;
        $('#extensions_settings').append(settingsHtml);
        $('#memory_restore').on('click', onMemoryRestoreClick);
        $('#memory_contents').on('input', onMemoryContentInput);
        $('#memory_long_length').on('input', onMemoryLongInput);
        $('#memory_short_length').on('input', onMemoryShortInput);
    }

    addExtensionControls();
    loadSettings();
    setInterval(moduleWorker, UPDATE_INTERVAL);
});