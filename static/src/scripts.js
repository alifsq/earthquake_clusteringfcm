// Mengatur active state saat link di-klik
document.querySelectorAll('nav a').forEach(link => {
    link.addEventListener('click', function () {
        // Menghapus kelas 'active' dari semua link
        document.querySelectorAll('nav a').forEach(item => item.classList.remove('active'));
        // Menambahkan kelas 'active' ke link yang di-klik
        this.classList.add('active');
    });
});

//----Modal Manual Add---
const modalManualAdd = document.getElementById('modalManualAdd')
const openModalManualAdd = document.getElementById('openManualAdd')
const closeModalManualAdd = document.getElementById('closeModalManualAdd')

// Open modal
openModalManualAdd.addEventListener('click', function () {
    modalManualAdd.classList.remove('hidden')
})

// Close modal
closeModalManualAdd.addEventListener('click', function () {
    modalManualAdd.classList.add('hidden')
})

// Close modal when clicking outside of the modal
window.onclick = function (event) {
    if (event.target === modalManualAdd) {
        modalManualAdd.classList.add('hidden')
    }
}

//Modal Edit data
function openEditModal(id, tanggal, waktu, latitude, longitude, magnitude, depth) {
    document.getElementById("editId").value = id;
    document.getElementById("editTanggal").value = tanggal;
    document.getElementById("editWaktu").value = waktu;
    document.getElementById("editLatitude").value = latitude;
    document.getElementById("editLongitude").value = longitude;
    document.getElementById("editMagnitude").value = magnitude;
    document.getElementById("editDepth").value = depth;
    document.getElementById("editModal").classList.remove("hidden");
}

function closeEditModal() {
    document.getElementById("editModal").classList.add("hidden");
}

//Modal drag and drop file csv
//Referencing HTML elements
const csvModal = document.getElementById("csvModal");
const openCsvAdd = document.getElementById("openCsvAdd");
const dropArea = document.getElementById("dropArea");
const fileInput = document.getElementById("fileInput");

// Buka modal saat tombol "Tambah Data CSV" diklik
openCsvAdd.addEventListener("click", function () {
    csvModal.classList.remove("hidden");
});

// Tutup modal
function closeCsvModal() {
    csvModal.classList.add("hidden");
}

// Prevent default drag behaviors
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// Highlight drop area when item is dragged over it
['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, () => {
        dropArea.classList.add('border-blue-500');
    }, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, () => {
        dropArea.classList.remove('border-blue-500');
    }, false);
});

// Handle dropped files
dropArea.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

function handleFiles(files) {
    // Assign dropped files to the file input
    fileInput.files = files;
    const fileName = files[0].name;
    dropArea.innerHTML = `<p class="text-gray-600">Selected File: ${fileName}</p>`;
}

// Optional: Preview file name when a file is selected via "Choose File"
fileInput.addEventListener("change", function () {
    const fileName = fileInput.files[0].name;
    dropArea.innerHTML = `<p class="text-gray-600">Selected File: ${fileName}</p>`;
});

setTimeout(function() {
    const flashMessage = document.querySelector('.flash-message');
    if (flashMessage) {
      flashMessage.classList.add('opacity-0');  // Menggunakan Tailwind untuk fade-out
      setTimeout(() => flashMessage.style.display = 'none', 300);  // Setelah fade-out selesai, sembunyikan
    }
  }, 3000);  // 5000ms = 5 detik
