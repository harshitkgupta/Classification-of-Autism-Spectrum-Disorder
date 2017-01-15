<?php
   require_once('loader.php');
   
    $resultUsers =  getAllUsers();
	if ($resultUsers != false)
		$NumOfUsers = mysql_num_rows($resultUsers);
	else
		$NumOfUsers = 0;
?>
<!DOCTYPE html>
<html>
    <head>
        <title></title>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.8.2/jquery.min.js"></script>
        <script type="text/javascript">
            $(document).ready(function(){
               
            });
            function sendPushNotification(id){
                var data = $('form#'+id).serialize();
                $('form#'+id).unbind('submit');                
                $.ajax({
                    url: "send_push_notification_message.php",
                    type: 'GET',
                    data: data,
                    beforeSend: function() {
                        
                    },
                    success: function(data, textStatus, xhr) {
                          $('.push_message').val("");
                    },
                    error: function(xhr, textStatus, errorThrown) {
                        
                    }
                });
                return false;
            }
        </script>
        <style type="text/css">
            
            h1{
                font-family:Helvetica, Arial, sans-serif;
                font-size: 24px;
                color: #777;
            }
            div.clear{
                clear: both;
            }
            
            textarea{
                float: left;
                resize: none;
            }
            
        </style>
    </head>
    <body>
        
        
        <table  width="910" cellpadding="1" cellspacing="1" style="padding-left:10px;">
         <tr>
           <td align="left">
              <h1>No of Devices Registered: <?php echo $NumOfUsers; ?></h1>
              <hr/>
           </td>
          </tr> 
          <tr>
            <td align="center">
              <table width="100%" cellpadding="1" cellspacing="1" style="border:1px solid #CCC;" bgcolor="#f4f4f4">
                <tr>
                  
               <?php
                if ($NumOfUsers > 0) {
                    $i=1;
                    while ($rowUsers = mysql_fetch_array($resultUsers)) {
						if($i%3==0)
						  print "</tr><tr><td colspan='2'>&nbsp;</td></tr><tr>";
                 ?>
                        <td align="left">
                             <form id="<?php echo $rowUsers["id"] ?>" name="" method="post" onSubmit="return sendPushNotification('<?php echo $rowUsers["id"] ?>')">
<label><b>Id:</b></label> <span><?php echo $rowUsers["id"] ?></span>
                                <div class="clear"></div>
                                <label><b>Name:</b> </label> <span><?php echo $rowUsers["name"] ?></span>
                                <div class="clear"></div>
                                <label><b>Email:</b></label> <span><?php echo $rowUsers["email"] ?></span>
<div class="clear"></div>
                                
                                <div class="send_container">                                
                                    <textarea rows="4" name="message" cols="80" class="push_message" placeholder="Type push message here"></textarea>
                                    <input type="hidden" name="regId" value="<?php echo $rowUsers["gcm_regid"] ?>"/>
                                    <input type="submit"  value="Send Push Notification" onClick=""/>
                                </div>
                            </form>
                         </td>
</tr>
                    <?php }
                } else { ?> 
                      <td>
                        User not exist.
                       </td>
                <?php } ?>
                    
                
                </table>
            </td>
          </tr>  
        </table>
        
        
    </body>
</html>