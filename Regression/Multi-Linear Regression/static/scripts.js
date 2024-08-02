document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('form');
    form.addEventListener('submit', function (event) {
        const inputs = form.querySelectorAll('input');
        let valid = true;

        inputs.forEach(input => {
            if (isNaN(input.value) || input.value.trim() === '') {
                valid = false;
                input.style.borderColor = 'red';
            } else {
                input.style.borderColor = '#ccc';
            }
        });

        if (!valid) {
            event.preventDefault();
            alert('Please enter valid numeric values.');
        }
    });
});
