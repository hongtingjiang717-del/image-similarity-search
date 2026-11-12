// 【初学者阅读说明】
// 浏览器执行本文件。HTML 提供元素，CSS 决定外观，JavaScript 监听点击、上传图片并显示结果。
// 这里不会训练 PyTorch 模型；真正计算发生在 Python 后端，通过 fetch 发送 HTTP 请求。
// const/let 声明变量，function/箭头函数定义动作，async/await 按顺序组织异步请求。
// DOM 就是浏览器中 HTML 的对象表示；通过元素对象修改 src、hidden、value 来更新页面。

// 开启严格模式，让某些含糊的 JavaScript 写法直接报错，例如给没有声明的变量赋值。
"use strict";
// const 定义不重新赋值的变量；document 是当前网页，getElementById 按唯一 id 找到文件选择框。
const input = document.getElementById("image-upload");
// 保存原图预览 <img> 元素的引用；修改其 src 属性就会改变显示的图片。
const preview = document.getElementById("image-preview");
// 找到展示模型实际输入图的 <img>，后面可在这里显示人为加噪的图片。
const noisy = document.getElementById("image-noised");
// 找到去噪结果 <img>；DOM 元素引用不等于图片二进制内容。
const denoised = document.getElementById("image-denoising");
// 找到状态提示段落；textContent 用来显示处理进度或错误，不把文本当 HTML 执行。
const status = document.getElementById("status");
// 找到只读分类结果文本框；后面通过 value 属性填入类别和模型分数。
const resultText = document.getElementById("result-text");
// 找到相似商品容器，查询成功后动态向其中添加图片元素。
const similar = document.getElementById("similar-images");
// let 声明以后会重新赋值的变量；null 表示还没有选中的文件。
let selectedFile = null;
// 保存去噪结果的二进制 Blob；有结果时分类/检索优先用它，否则用原图。
let denoisedBlob = null;
// 保存浏览器为本地文件生成的临时 URL，换图时要释放以免占用内存。
let previewUrl = null;
// 布尔开关：true 表示请求尚未结束，用来防止用户同时触发多次处理。
let busy = false;

// 定义清空结果函数；函数只是登记一段操作，调用它时才执行函数体。
function resetResults() {
    // 清除上一张图片的恢复结果，避免新图片误用旧结果分类或检索。
    denoisedBlob = null;
    // for...of 依次取数组中的两个图片元素，统一清空它们。
    for (const image of [noisy, denoised]) {
        // hidden=true 使元素暂时不显示，避免页面残留上次的结果。
        image.hidden = true;
        // 删除旧图片地址，不再指向上次处理的图片。
        image.removeAttribute("src");
    }
    // 将分类结果文本框内容设置为空字符串。
    resultText.value = "";
    // 不传子节点时清空容器，去掉之前显示的相似商品图片。
    similar.replaceChildren();
    // 清空旧的进度或错误提示。
    status.textContent = "";
}

// 处理选图动作：更新当前文件、清理旧结果，并创建新预览。
function selectFile() {
    // 请求进行中则提前返回；return 在这里结束当前函数，避免重复动作。
    if (busy) return;
    // files 是文件列表；[0] 取第一项，|| 在没有文件时使用 null。
    selectedFile = input.files[0] || null;
    // 调用前面定义的清理函数，让界面与当前图片保持一致。
    resetResults();
    // 如果有旧临时 URL 就释放它；对象 URL 在不再使用时应交还浏览器。
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    // 清除已经失效的 URL 记录，不再误用旧地址。
    previewUrl = null;
    // 在确认有新图片之前先隐藏预览区域中的图片元素。
    preview.hidden = true;
    // 移除旧预览地址，避免换图取消时仍显示上一张。
    preview.removeAttribute("src");
    // 只有实际选择了文件时，才进入下一步创建预览。
    if (selectedFile) {
        // 在浏览器本地生成临时访问地址；此时没有把文件上传到服务器。
        previewUrl = URL.createObjectURL(selectedFile);
        // 把临时 URL 填入图片的 src，浏览器据此显示本地选图。
        preview.src = previewUrl;
        // 取消隐藏，展示新的原图预览。
        preview.hidden = false;
    }
}
// 给文件框注册 change 事件：用户更换文件后自动调用 selectFile；这里传函数本身，不写 selectFile()。
input.addEventListener("change", selectFile);
// 箭头函数 () => {...} 登记上传按钮点击时的动作；设置 onclick 不会立即执行动作。
document.getElementById("upload_button").onclick = () => {
    // ! 表示取反；尚无文件时，程序触发文件选择框，让用户选图。
    if (!input.files.length) input.click();
    // 已经选过文件时，直接调用选图处理函数刷新预览；这里的“上传”按钮实际先完成本地预览。
    else selectFile();
};

// async 定义返回 Promise 的异步函数；它负责把图片上传并统一解析后端 JSON。
async function requestImage(endpoint, blob, extra = {}) {
    // 创建表单容器，支持同时放文件和普通字符串字段。
    const form = new FormData();
    // 添加名为 image 的文件字段，必须与后端 request.files.get("image") 对应；第三个参数是上传文件名。
    form.append("image", blob, "image.png");
    // Object.entries 把对象转成键值对；[key,value] 是解构赋值；循环把额外选项加到同一个表单。
    for (const [key, value] of Object.entries(extra)) form.append(key, value);
    // fetch 发 HTTP 请求；POST 携带图片表单；await 等待网络响应，不阻塞整个浏览器界面。
    const response = await fetch(endpoint, { method: "POST", body: form });
    // 等待响应体读完并按 JSON 解析成 JavaScript 对象；字段应与 Python jsonify 返回的键对应。
    const data = await response.json();
    // HTTP 4xx/5xx 不会让 fetch 自动抛异常，所以主动检查 ok，并把后端错误变成可捕获的 Error。
    if (!response.ok) throw new Error(data.error || "请求失败");
    // 把已解析的结果对象返回给调用者；async 会把它包装在 Promise 中。
    return data;
}

// 统一处理按钮请求：检查图片、锁定控件、执行动作、展示错误并最终恢复界面。action 是传进来的异步函数。
async function perform(action) {
    // 请求进行中则提前返回；return 在这里结束当前函数，避免重复动作。
    if (busy) return;
    // 还没选图就给出提示并提前返回；不能让后端收到没有图片的业务请求。
    if (!selectedFile) { status.textContent = "请先选择图片。"; return; }
    // 标记正在处理，从现在起其他按钮请求会被前面的 busy 检查拦住。
    busy = true;
    // querySelectorAll 选中所有按钮和输入框；... 把节点列表展开成普通数组。
    const controls = [...document.querySelectorAll("button, input")];
    // forEach 遍历控件；箭头函数将每个 disabled 设为 true，请求期间禁止改图或重复提交。
    controls.forEach(control => control.disabled = true);
    // 显示当前处理进度；这条文字不代表模型已经完成计算。
    status.textContent = "正在处理，请稍候……";
    // try 中执行可能失败的网络操作；异常会跳到后面的 catch。
    try {
        // 真正运行调用方传来的任务，并等待它完成后再继续更新状态。
        await action();
        // 只有 action 成功结束后才显示完成信息。
        status.textContent = "处理完成。";
    // 捕获错误对象，例如 HTTP 失败或网络中断；不会把错误误显示为成功。
    } catch (error) {
        // 把错误的可读 message 显示在页面中，用户无需打开开发者工具才能看到。
        status.textContent = error.message;
    // finally 无论成功还是失败都会执行，确保按钮不会一直卡在禁用状态。
    } finally {
        // 解除请求进行中的标记，允许下一次操作。
        busy = false;
        // 恢复控件，用户现在可以换图或再次处理。
        controls.forEach(control => control.disabled = false);
    }
}

// 登记去噪按钮事件；外层用 perform 管理状态，内层 async 函数定义具体去噪步骤。
document.getElementById("denoising_button").onclick = () => perform(async () => {
    // 调用前面定义的清理函数，让界面与当前图片保持一致。
    resetResults();
    // 清空旧结果后重新显示去噪进度。
    status.textContent = "正在去噪，请稍候……";
    // 将当前原始文件上传到去噪路由；额外表单字段决定是否先人为加噪。
    const data = await requestImage("/denoising", selectedFile, {
        // checked 是复选框布尔值；条件表达式把它转成后端约定的字符串 true 或 false。
        add_noise: document.getElementById("add-noise").checked ? "true" : "false"
    });
    // 模板字符串使用反引号，${...} 插入变量；前缀说明后面的文本是 Base64 PNG。
    noisy.src = `data:image/png;base64,${data.noisy_img}`;
    // 把后端恢复图的 Base64 数据写成 data URL，供浏览器显示。
    denoised.src = `data:image/png;base64,${data.denoised_image}`;
    // 链式赋值：两个图片元素都取消隐藏，不是比较它们是否相等。
    noisy.hidden = denoised.hidden = false;
    // 先读取 data URL，再转成二进制 Blob，供后续分类与检索作为上传文件使用。
    denoisedBlob = await (await fetch(denoised.src)).blob();
});
// 登记商品分类按钮点击动作，复用同一套请求状态管理。
document.getElementById("classification_button").onclick = () => perform(async () => {
    // 如果已有去噪结果就上传它，否则上传选中的原图；|| 在左边无结果时才采用右边。
    const data = await requestImage("/classification", denoisedBlob || selectedFile);
    // 在模板字符串中插入类别名和百分数；乘 100 转百分比，toFixed(1) 保留一位小数。
    resultText.value = `商品类型：${data.name}；模型置信度：${(data.confidence * 100).toFixed(1)}%`;
});
// 登记相似商品按钮事件，收到结果后在页面动态创建商品图片。
document.getElementById("similarity_button").onclick = () => perform(async () => {
    // 上传去噪结果或原图，向后端请求相似商品记录。
    const data = await requestImage("/simimages", denoisedBlob || selectedFile);
    // 不传子节点时清空容器，去掉之前显示的相似商品图片。
    similar.replaceChildren();
    // 逐项遍历后端 results 数组；每个 item 有 filename、distance、url 等字段。
    for (const item of data.results) {
        // 创建新的 <img> 节点；刚创建时还未加入页面。
        const image = document.createElement("img");
        // 直接采用后端生成的真实图片 URL，不再拼接“序号.jpg”猜测文件名。
        image.src = item.url;
        // 设置替代文本；图片不能显示或使用辅助阅读工具时仍能知道文件名。
        image.alt = item.filename;
        // 设置鼠标悬停提示，展示文件名与余弦距离；距离越小表示当前特征定义下越相似。
        image.title = `${item.filename}，余弦距离 ${item.distance.toFixed(4)}`;
        // 把刚创建的图片节点加入结果容器，它现在才真正出现在页面结构里。
        similar.appendChild(image);
    }
});
