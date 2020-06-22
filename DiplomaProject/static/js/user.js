$("button[name='btn_delete_user']").click(function() {

    var data = { user_id : $(this).data('user_id')}

    $.ajax({
      type: 'POST',
      url: "/delete_user",
      data: data,
      dataType: "text",
      success: function(resultData) {
          location.reload();
      }
});
});


$("button[name='btn_edit_user']").click(function() {

    window.location = "edit_user?user_id="+$(this).data('user_id');

});


$("button[name='btn_new_user']").click(function() {

    window.location = "new_user";

});

