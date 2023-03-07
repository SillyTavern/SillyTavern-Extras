const MODULE_NAME = 'caption';
const UPDATE_INTERVAL = 1000;

const getContext = function () {
    return window['TavernAI'].getContext();
}

const getApiUrl = function () {
    return localStorage.getItem('extensions_url');
}

function getBase64Async(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = function () {
            resolve(reader.result);
        };
        reader.onerror = function (error) {
            reject(error);
        };
    });
}

async function moduleWorker() {
    const context = getContext();

    context.onlineStatus === 'no_connection'
        ? $('#send_picture').hide(200)
        : $('#send_picture').show(200);
}

async function urlContentToDataUri(url, params) {
    const response = await fetch(url, params);
    const blob = await response.blob();
    return await new Promise(callback => {
        let reader = new FileReader();
        reader.onload = function () { callback(this.result); };
        reader.readAsDataURL(blob);
    });
}

async function setImageIcon() {
    try {
        const sendButton = document.getElementById('send_picture');
        const imgUrl = new URL(getApiUrl());
        imgUrl.pathname = `/api/asset/${MODULE_NAME}/image-solid.svg`;
        const dataUri = await urlContentToDataUri(imgUrl.toString(), { method: 'GET', headers: { 'Bypass-Tunnel-Reminder': 'bypass' } });
        sendButton.style.backgroundImage = `url(${dataUri})`;
        sendButton.classList.remove('spin');
    }
    catch (error) {
        console.log(error);
    }
}

async function setSpinnerIcon() {
    try {
        const sendButton = document.getElementById('send_picture');
        const imgUrl = new URL(getApiUrl());
        imgUrl.pathname = `/api/asset/${MODULE_NAME}/spinner-solid.svg`;
        const dataUri = await urlContentToDataUri(imgUrl.toString(), { method: 'GET', headers: { 'Bypass-Tunnel-Reminder': 'bypass' } });
        sendButton.style.backgroundImage = `url(${dataUri})`;
        sendButton.classList.add('spin');
    }
    catch (error) {
        console.log(error);
    }
}

async function sendCaptionedMessage(caption, image) {
    const context = getContext();
    const messageText = `[${context.name1} sends ${context.name2 ?? ''} a picture that contains: ${caption}]`;
    const message = {
        name: context.name1,
        is_user: true,
        is_name: true,
        send_date: Date.now(),
        mes: messageText,
        extra: { image: image },
    };
    context.chat.push(message);
    context.addOneMessage(message);
    await context.generate();
}

async function onSelectImage(e) {
    setSpinnerIcon();
    const file = e.target.files[0];

    if (!file) {
        return;
    }

    try {
        const base64Img = await getBase64Async(file);
        const url = new URL(getApiUrl());
        url.pathname = '/api/caption';

        const apiResult = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Bypass-Tunnel-Reminder': 'bypass',
            },
            body: JSON.stringify({ image: base64Img.split(',')[1] })
        });

        if (apiResult.ok) {
            const data = await apiResult.json();
            const caption = data.caption;
            await sendCaptionedMessage(caption, base64Img);
        }
    }
    catch (error) {
        console.log(error);
    }
    finally {
        e.target.form.reset();
        setImageIcon();
    }
}

$(document).ready(function () {
    function patchSendForm() {
        const columns = $('#send_form').css('grid-template-columns').split(' ');
        columns[columns.length - 1] = `${parseInt(columns[columns.length - 1]) + 40}px`;
        columns[1] = 'auto';
        $('#send_form').css('grid-template-columns', columns.join(' '));
    }
    function addSendPictureButton() {
        const sendButton = document.createElement('input');
        sendButton.type = 'button';
        sendButton.id = 'send_picture';
        $(sendButton).hide();
        $(sendButton).on('click', () => $('#img_file').click());
        $('#send_but_sheld').prepend(sendButton);
    }
    function addPictureSendForm() {
        const inputHtml = `<input id="img_file" type="file" accept="image/*">`;
        const imgForm = document.createElement('form');
        imgForm.id = 'img_form';
        $(imgForm).append(inputHtml);
        $(imgForm).hide();
        $('#form_sheld').append(imgForm);
        $('#img_file').on('change', onSelectImage);
    }

    addPictureSendForm();
    addSendPictureButton();
    setImageIcon();
    patchSendForm();
    setInterval(moduleWorker, UPDATE_INTERVAL);
});