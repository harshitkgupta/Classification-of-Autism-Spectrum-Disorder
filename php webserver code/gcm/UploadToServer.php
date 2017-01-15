<?php
require_once('loader.php');
$user = $_REQUEST['user'];
$q = mysql_query("SELECT * FROM gcm_users WHERE gcm_regid = '$user' ");
$row = mysql_fetch_array($q);
$email=$row['email'];
$target_path1 = "uploads/".$email."/";
if(!is_dir($target_path1)) 
		{
			    mkdir($target_path1);         
		}
/* Add the original filename to our target path.
Result is "uploads/filename.extension" */
$target_path1 = $target_path1 . basename( $_FILES['uploadedfile1']['name']);
if(move_uploaded_file($_FILES['uploadedfile1']['tmp_name'], $target_path1)) {
    echo "\n The audio file ".  basename( $_FILES['uploadedfile1']['name']). " has been uploaded.";
} else{
    echo "There was an error uploading the file, please try again!";
    echo "filename: " .  basename( $_FILES['uploadedfile1']['name']);
    echo "target_path: " .$target_path1;
}
 

 

echo "\n your file has been uploaded to your user directory : " . $target_path1;
//$registatoin_ids = array($user);
//$message = array("price" => 'thanks for uploading speech samples');
//send_push_notification($registatoin_ids, $message);
//$message = array("price" => 'We will notify you with the results as soon as we process result');
//send_push_notification($registatoin_ids, $message);
?>