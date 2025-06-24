import { execSync, spawn } from "child_process";

export const execCommand = (command: string): string => {
  try {
    return execSync(command).toString().trim();
  } catch (error) {
    console.error(`Error executing command: ${command}`);
    console.error(error);
    throw error;
  }
};

export const execCommandInBackground = (
  command: string,
  args: string[]
): number | undefined => {
  const child = spawn(command, args, { detached: true });

  child.stdout.on("data", (data) => {
    console.log(`stdout (${command}): ${data}`);
  });
  child.stderr.on("data", (data) => {
    console.log(`stderr (${command}): ${data}`);
  });
  child.on("close", (code) => {
    console.log(`child process (${command}) exited with code ${code}`);
  });
  child.on("error", (error) => {
    console.error(`child process (${command}) error: ${error}`);
    throw error;
  });

  return child.pid;
};

export const killProcess = (pid: number) => {
  process.kill(pid);
};

export const sleep = async (seconds: number) => {
  return new Promise((resolve) => {
    setTimeout(resolve, seconds * 1000);
  });
};

export const downloadFile = async (url: string, destination: string) => {
  const path = require("path");
  const https = require("https");
  const http = require("http");
  const fs = require("fs");

  const dir = path.dirname(destination);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  return new Promise((resolve, reject) => {
    const downloadWithRedirect = (downloadUrl: string, maxRedirects = 5) => {
      const client = downloadUrl.startsWith("https:") ? https : http;

      const request = client.get(downloadUrl, (response: any) => {
        // Handle redirects
        if (
          response.statusCode >= 300 &&
          response.statusCode < 400 &&
          response.headers.location
        ) {
          if (maxRedirects > 0) {
            console.log(`Redirecting to: ${response.headers.location}`);
            return downloadWithRedirect(
              response.headers.location,
              maxRedirects - 1
            );
          } else {
            return reject(new Error("Too many redirects"));
          }
        }

        // Check for successful response
        if (response.statusCode !== 200) {
          return reject(
            new Error(`HTTP ${response.statusCode}: ${response.statusMessage}`)
          );
        }

        // Check content length
        const contentLength = parseInt(
          response.headers["content-length"] || "0"
        );
        console.log(`Downloading file, expected size: ${contentLength} bytes`);

        const file = fs.createWriteStream(destination);
        let downloadedBytes = 0;

        response.on("data", (chunk: any) => {
          downloadedBytes += chunk.length;
        });

        response.pipe(file);

        file.on("finish", () => {
          file.close((err: any) => {
            if (err) {
              return reject(err);
            }

            console.log(`File downloaded successfully to ${destination}`);
            console.log(`Downloaded ${downloadedBytes} bytes`);

            // Verify file size
            const stats = fs.statSync(destination);
            if (stats.size === 0) {
              return reject(new Error("Downloaded file is empty"));
            }

            // Give permissions to the file
            fs.chmod(destination, 0o644, (chmodErr: any) => {
              if (chmodErr) {
                console.error(
                  `Error setting permissions for ${destination}:`,
                  chmodErr
                );
                return reject(chmodErr);
              }
              console.log(`Permissions set for ${destination}`);
              resolve(true);
            });
          });
        });

        file.on("error", (err: any) => {
          fs.unlink(destination, () => reject(err));
        });
      });

      request.on("error", (err: any) => {
        reject(err);
      });

      request.setTimeout(30000, () => {
        request.destroy();
        reject(new Error("Download timeout"));
      });
    };

    downloadWithRedirect(url);
  });
};
