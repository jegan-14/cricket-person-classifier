$(document).ready(function () {
    $('.error').hide();
    $('.classified').hide();
    classify();
})

let dataURL = null;

function classify() {
    const btn = document.getElementById('btn');
    const fileInput = document.getElementById("fileInput");

    btn.addEventListener("click", () => {
        if (fileInput.files.length > 0) {
            const img = fileInput.files[0];
            const reader = new FileReader();

            reader.addEventListener("load", () => {
                dataURL = reader.result;
                // Move the AJAX request inside the load event
                var api = "http://127.0.0.1:5000/classify_image";
                $.post(
                    api, {
                        image_data: dataURL
                    }, function (data, status) {
                        console.log(data);
                        
                        $('.classified').hide(); 
                        if(data.length === 0){
                            $('.error').show();
                            $('.classified').hide(); 
                        }else{
                        
                        console.log(data[0].class)
                        var cls = data[0].class;
                        $('.'+cls).show();
                        $('.error').hide(); 
                        }
                    }
                    
                );
            });

            reader.readAsDataURL(img);
        } else {
            console.log("Please select a file before clicking the button.");
        }
    });
}

