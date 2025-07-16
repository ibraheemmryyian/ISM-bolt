const axios = require('axios');

/**
 * Universal HTTP request with retry, timeout, and header propagation
 * @param {Object} config - Axios config (method, url, headers, data, timeout, etc.)
 * @param {number} retries - Number of retry attempts
 * @param {number} retryDelay - Initial delay in ms for exponential backoff
 * @returns {Promise<AxiosResponse>}
 */
async function requestWithRetry(config, retries = 3, retryDelay = 1000) {
  let attempt = 0;
  let lastError;
  while (attempt <= retries) {
    try {
      return await axios(config);
    } catch (error) {
      lastError = error;
      if (attempt === retries) break;
      // Only retry on network/server errors
      if (!error.response || error.response.status >= 500) {
        await new Promise(res => setTimeout(res, retryDelay * Math.pow(2, attempt)));
        attempt++;
      } else {
        break;
      }
    }
  }
  throw lastError;
}

module.exports = requestWithRetry; 